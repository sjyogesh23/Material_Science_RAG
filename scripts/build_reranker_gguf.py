"""Build the Ollama reranker model ``qwen3-reranker:0.6b-q8_0`` from the HF Qwen/Qwen3-Reranker-0.6B weights.

Why: Ollama 0.40 cannot import safetensors on Windows (needs MLX) and the public
``dengcao/Qwen3-Reranker-0.6B:Q8_0`` upload has all-zero weights (outputs ",,,,"). Qwen3-Reranker-0.6B has the
exact architecture + tokenizer of Qwen3-Embedding-0.6B, so we take Ollama's ``qwen3-embedding:0.6b`` GGUF as the
metadata/tokenizer template (minus ``pooling_type`` -> causal LM with the tied output head), quantise the
reranker's bf16 tensors to Q8_0 and ``ollama create`` it. No torch needed (~1.5 GB RAM peak).

    .venv/Scripts/python scripts/build_reranker_gguf.py            # needs: pip install gguf; ollama pull qwen3-embedding:0.6b
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_MODELS = Path(os.getenv("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
OLLAMA_EXE = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
LAYER_MAP = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
             "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
             "attn_k_norm": "self_attn.k_norm", "ffn_norm": "post_attention_layernorm",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


class SafeTensors:
    """Minimal torch-free safetensors reader (BF16 / F32 -> float32 numpy)."""

    def __init__(self, path: Path):
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, key: str) -> np.ndarray:
        meta = self.header[key]
        a, b = meta["data_offsets"]
        raw = self.mm[self.base + a:self.base + b]
        if meta["dtype"] == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
        if meta["dtype"] == "BF16":
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
        raise ValueError(f"unsupported dtype {meta['dtype']} for {key}")


def ollama_blob(model: str, media: str = "application/vnd.ollama.image.model") -> Path:
    name, tag = (model.split(":", 1) + ["latest"])[:2]
    parts = name.split("/")
    ns, repo = (parts[0], parts[1]) if len(parts) == 2 else ("library", parts[0])
    manifest = OLLAMA_MODELS / "manifests" / "registry.ollama.ai" / ns / repo / tag
    layers = json.loads(manifest.read_text())["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"] == media)
    return OLLAMA_MODELS / "blobs" / digest.replace(":", "-")


def hf_snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=["model.safetensors", "config.json"]))


def hf_name(gguf_name: str) -> str:
    base = gguf_name.removesuffix(".weight")
    if base == "token_embd":
        return "model.embed_tokens.weight"
    if base == "output_norm":
        return "model.norm.weight"
    _, idx, part = base.split(".", 2)
    return f"model.layers.{idx}.{LAYER_MAP[part]}.weight"


def build(template: Path, weights: Path, out: Path) -> None:
    import gguf
    from gguf import GGMLQuantizationType as Q, GGUFReader, GGUFValueType as VT, GGUFWriter
    from gguf.quants import quantize

    reader, st = GGUFReader(template), SafeTensors(weights)
    w = GGUFWriter(str(out), "qwen3")
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "qwen3.pooling_type",
            "tokenizer.ggml.add_eos_token", "general.name", "general.basename", "general.file_type", "general.tags"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        vt = field.types[0]
        if vt == VT.ARRAY:
            w.add_key_value(key, field.contents(), VT.ARRAY, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vt)
    w.add_name("Qwen3 Reranker 0.6B")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    w.add_bool("tokenizer.ggml.add_eos_token", False)
    for t in reader.tensors:
        arr = st.get(hf_name(t.name)).astype(np.float32)
        if t.tensor_type == Q.F32:
            w.add_tensor(t.name, arr)
        else:
            q = quantize(arr, Q.Q8_0)
            w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=Q.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template-model", default="qwen3-embedding:0.6b")
    ap.add_argument("--hf-repo", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--name", default="qwen3-reranker:0.6b-q8_0")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "cache" / "reranker-build")
    ap.add_argument("--keep-gguf", action="store_true", help="keep the .gguf after `ollama create` (640 MB)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = args.out_dir / "Qwen3-Reranker-0.6B-Q8_0.gguf"
    build(ollama_blob(args.template_model), hf_snapshot(args.hf_repo) / "model.safetensors", gguf_path)
    modelfile = args.out_dir / "Modelfile"
    modelfile.write_text(f'FROM ./{gguf_path.name}\nTEMPLATE """{{{{ .Prompt }}}}"""\n'
                         "PARAMETER temperature 0\nPARAMETER num_ctx 1024\n", encoding="utf-8")
    exe = str(OLLAMA_EXE) if OLLAMA_EXE.exists() else "ollama"
    subprocess.run([exe, "create", args.name, "-f", str(modelfile)], cwd=args.out_dir, check=True)
    if not args.keep_gguf:
        gguf_path.unlink(missing_ok=True)
    print(f"created Ollama model {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
