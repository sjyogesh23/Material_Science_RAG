"""Vision-language model backends for figure descriptions.

* ``local`` – a Hugging Face image-text-to-text model run in-process (default Qwen3.5-0.8B,
  ~1.8 GB VRAM in fp16; larger Qwen3.5 models can be loaded in 4-bit).
* ``api``   – any OpenAI-compatible chat endpoint (Ollama, LM Studio, vLLM, llama.cpp server ...).

The model is asked for strict JSON; output is parsed defensively.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re

from PIL import Image

from .config import Settings

log = logging.getLogger(__name__)

IMAGE_TYPES = ("flow_chart|schematic|diagram|SEM_micrograph|TEM_micrograph|optical_micrograph|AFM_image|"
               "EBSD_map|XRD_pattern|spectrum|phase_diagram|stress_strain_curve|line_chart|bar_chart|"
               "scatter_plot|heatmap|contour_map|crystal_structure|molecular_structure|simulation_snapshot|"
               "photo|equipment_photo|sample_photo|table_image|equation_image|logo|signature|stamp|other")

GENERAL_PROMPT = (
    "You are analysing a figure from a materials-science document.{ctx} "
    "Return ONLY compact JSON with these keys:\n"
    '{{"description": "<3-5 factual sentences: what is shown, layout, colours, labels, and the key scientific message>", '
    '"short_caption": "<max 12 words>", '
    f'"image_type": "<one of {IMAGE_TYPES}>", '
    '"objects": [{{"label": "", "color": ""}}], '
    '"materials": ["<materials or samples shown, if any>"], '
    '"technique": "<characterisation / simulation technique if identifiable, else null>"}}'
)

CHART_PROMPT = (
    "You are analysing a scientific chart from a materials-science document.{ctx} "
    "Return ONLY compact JSON with these keys:\n"
    '{{"description": "<3-5 factual sentences describing what is plotted and the main trends/findings>", '
    '"short_caption": "<max 12 words>", '
    '"chart_type": "<line|bar|stacked_bar|scatter|pie|box|histogram|heatmap|contour|XRD_pattern|spectrum|'
    'stress_strain_curve|phase_diagram|other>", '
    '"title": "<chart title or null>", '
    '"x_axis": {{"label": "", "unit": "", "min": null, "max": null, "scale": "linear|log"}}, '
    '"y_axis": {{"label": "", "unit": "", "min": null, "max": null, "scale": "linear|log"}}, '
    '"series": [{{"name": "", "color": "", "trend": "<increasing|decreasing|peak|constant|...>"}}], '
    '"legend": [""], '
    '"key_findings": ["<short quantitative observations, e.g. peak at 2θ≈44°>"]}}'
)

CHART_CLASSES = {"bar_chart", "line_chart", "pie_chart", "scatter_plot", "scatter_chart", "box_plot",
                 "other_chart", "stacked_bar_chart", "heatmap", "chart"}


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from model output, repairing common small-model mistakes."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, end = 0, None
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    cand = text[start:end] if end else text[start:] + "}" * depth
    for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    # last resort: grab the description string
    m = re.search(r'"description"\s*:\s*"([^"]+)', text)
    return {"description": m.group(1)} if m else None


class VLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.backend = settings.vlm_backend.lower()
        self.model_name = settings.vlm_local_model if self.backend == "local" else settings.vlm_api_model
        self._model = self._proc = None

    # ------------------------------------------------------------------ local
    def _load_local(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self.s.resolved_device
        kwargs: dict = {"device_map": device if device != "cpu" else None}
        if self.s.vlm_load_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32
        log.info("loading VLM %s on %s", self.s.vlm_local_model, device)
        self._proc = AutoProcessor.from_pretrained(self.s.vlm_local_model)
        self._model = AutoModelForImageTextToText.from_pretrained(self.s.vlm_local_model, **kwargs).eval()

    def _generate_local(self, img: Image.Image, prompt: str) -> str:
        import torch

        if self._model is None:
            self._load_local()
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        try:
            inp = self._proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                 return_tensors="pt", enable_thinking=False)
        except TypeError:  # processors without the enable_thinking switch
            inp = self._proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                 return_tensors="pt")
        inp = inp.to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inp, max_new_tokens=self.s.vlm_max_new_tokens, do_sample=False,
                                       repetition_penalty=1.05)
        return self._proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    # -------------------------------------------------------------------- api
    def _generate_api(self, img: Image.Image, prompt: str) -> str:
        import requests

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        headers = {"Content-Type": "application/json"}
        if self.s.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.s.vlm_api_key}"
        payload = {"model": self.s.vlm_api_model, "temperature": 0, "max_tokens": self.s.vlm_max_new_tokens,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
        r = requests.post(self.s.vlm_api_url, json=payload, headers=headers, timeout=self.s.vlm_timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ----------------------------------------------------------------- public
    def describe(self, img: Image.Image, picture_class: str | None = None, caption: str | None = None) -> dict:
        """Return a dict with description fields (+ chart fields for charts)."""
        img = img.convert("RGB")
        img.thumbnail((self.s.vlm_max_image_side, self.s.vlm_max_image_side))
        ctx = f' Its caption in the document is: "{caption[:300]}".' if caption else ""
        is_chart = (picture_class or "") in CHART_CLASSES
        prompt = (CHART_PROMPT if is_chart else GENERAL_PROMPT).format(ctx=ctx)
        try:
            raw = self._generate_local(img, prompt) if self.backend == "local" else self._generate_api(img, prompt)
        except Exception as e:  # noqa: BLE001
            log.warning("VLM failed: %s", e)
            return {"error": str(e)}
        parsed = parse_json(raw) or {"description": raw.strip()[:2000], "parse_error": True}
        parsed["_prompt_kind"] = "chart" if is_chart else "general"
        return parsed

    def unload(self):
        if self._model is not None:
            import gc

            import torch

            del self._model, self._proc
            self._model = self._proc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
