#!/usr/bin/env python3
"""Download open materials-science PDFs into ./data for a RAG knowledge base.

Usage (from the project root, with the venv active):
    python scripts/scrap.py                   # saves to <project>/data
    python scripts/scrap.py --out other_dir   # custom folder
    python scripts/scrap.py --max-per-page 40

Output:
    data/<category>/<id>_<title>[_n].pdf
    data/manifest.json   (source URL, license note, local path, sha256 for every file)

Re-running skips files that already exist. Respect each source's license.
"""
import argparse, hashlib, json, os, re, time
from pathlib import Path
from urllib.parse import urljoin

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HDRS = {"User-Agent": "Mozilla/5.0 (materials-kb-builder)"}
DELAY = 1.5  # seconds between requests, be polite

# (id, title, category, url, mode, license_note)
# modes: direct_pdf | pmc | scrape (find PDF links on the page)
SOURCES = [
    (1, "Steimel Materials Science and Engineering", "textbooks", "https://scholarlycommons.pacific.edu/open-textbooks/8", "scrape", "CC BY-NC-ND 4.0"),
    (5, "Sadoway Intro to Solid State Chemistry", "textbooks", "https://batch.libretexts.org/print/Letter/Finished/chem-408239/Full.pdf", "direct_pdf", "LibreTexts OER"),
    (6, "MIT OCW 3.012 Fundamentals of Materials Science", "mit_ocw", "https://ocw.mit.edu/courses/3-012-fundamentals-of-materials-science-fall-2005/", "scrape", "CC BY-NC-SA"),
    (7, "MIT OCW 3.016 Math for Materials Scientists", "mit_ocw", "https://ocw.mit.edu/courses/materials-science-and-engineering/3-016-mathematics-for-materials-scientists-and-engineers-fall-2005/", "scrape", "CC BY-NC-SA"),
    (8, "MIT OCW 3.225 Electronic and Mechanical Properties", "mit_ocw", "https://ocw.mit.edu/courses/3-225-electronic-and-mechanical-properties-of-materials-fall-2007/resources/lecture-notes", "scrape", "CC BY-NC-SA"),
    (9, "MIT OCW 3.46 Photonic Materials and Devices", "mit_ocw", "https://ocw.mit.edu/courses/materials-science-and-engineering/3-46-photonic-materials-and-devices-spring-2006/download-course-materials/", "scrape", "CC BY-NC-SA"),
    (10, "MIT OCW 3.23 Electrical Optical Magnetic Properties", "mit_ocw", "https://ocw.mit.edu/courses/3-23-electrical-optical-and-magnetic-properties-of-materials-fall-2007/resources/assignments", "scrape", "CC BY-NC-SA"),
    (11, "MIT OCW 3.091SC Solid State Chemistry", "mit_ocw", "https://ocw.mit.edu/courses/3-091sc-introduction-to-solid-state-chemistry-fall-2010/", "scrape", "CC BY-NC-SA"),
    (12, "DOE Material Science Handbook Vol 1", "handbooks", "https://www.standards.doe.gov/standards-documents/1000/1017-BHdbk-1993-v1", "scrape", "US Government work"),
    (13, "DOE Material Science Handbook Vol 2", "handbooks", "https://www.standards.doe.gov/standards-documents/1000/1017-BHdbk-1993-V2", "scrape", "US Government work"),
    (14, "PNNL-17764 Materials Science Teachers Handbook", "handbooks", "https://ceramics.org/wp-content/uploads/2024/08/PNNL-17764.pdf", "direct_pdf", "US Government sponsored"),
    (15, "NIST SP 960-9 Surface Engineering Standards", "handbooks", "https://www.nist.gov/node/399306", "scrape", "US Government work"),
    (17, "IIT Kanpur E-Book Materials Science", "textbooks", "https://home.iitk.ac.in/~anandh/E-book.htm", "scrape", "license not stated - verify"),
    (19, "Review Polymer Composites 3D-4D Printing", "reviews", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10818632/", "pmc", "MDPI open access"),
    (20, "Review Ceramic Nanocomposites Preceramic Polymers", "reviews", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5312884/", "pmc", "MDPI open access"),
    (21, "Review Self-Healing Composites", "reviews", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9740628/", "pmc", "open access"),
    (22, "Review Polymer-Nanoparticle Composites", "reviews", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5521756/", "pmc", "open access"),
    (23, "Review AM Highly-Filled Polymers", "reviews", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5978217/", "pmc", "MDPI open access"),
    (24, "Review Metal and Polymer Composites AM", "reviews", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10255547/", "pmc", "open access"),
    (25, "arXiv 2311.10205 Combinatorial Materials Data Management", "preprints", "https://arxiv.org/pdf/2311.10205", "direct_pdf", "arXiv per-paper license"),
]

session = requests.Session()
session.headers.update(HDRS)


def fetch(url):
    last = None
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def slug(s, n=100):
    return re.sub(r"[^\w.-]+", "_", s).strip("_")[:n]


def find_pdf_links(page_url):
    html = fetch(page_url).text
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    links = [urljoin(page_url, h) for h in hrefs
             if re.search(r"\.pdf($|\?)|viewcontent\.cgi", h, flags=re.I)]
    seen, out = set(), []
    for l in links:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def download(url, path):
    r = fetch(url)
    if not r.content.lstrip()[:5].startswith(b"%PDF"):
        return None
    with open(path, "wb") as f:
        f.write(r.content)
    return hashlib.sha256(r.content).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--max-per-page", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    done = {m["local_path"] for m in manifest}

    ok = fail = 0
    for sid, title, cat, url, mode, lic in SOURCES:
        folder = os.path.join(args.out, cat)
        os.makedirs(folder, exist_ok=True)
        try:
            if mode == "direct_pdf":
                targets = [url]
            elif mode == "pmc":
                targets = [url.rstrip("/") + "/pdf/"]
            else:
                targets = find_pdf_links(url)[: args.max_per_page]
                if not targets:
                    print(f"[no-links] {sid} {title}")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {sid} {title}: {e}")
            fail += 1
            continue

        for i, t in enumerate(targets):
            suffix = "" if len(targets) == 1 else f"_{i:02d}"
            path = os.path.join(folder, f"{sid}_{slug(title)}{suffix}.pdf")
            if path in done and os.path.exists(path):
                continue
            try:
                digest = download(t, path)
                if digest:
                    manifest.append({"id": sid, "title": title, "category": cat,
                                     "source_page": url, "pdf_url": t,
                                     "license": lic, "local_path": path,
                                     "sha256": digest})
                    ok += 1
                    print(f"[ok] {path}")
                else:
                    print(f"[not-a-pdf] {t}")
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"[fail] {t}: {e}")
            time.sleep(DELAY)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone. {ok} downloaded, {fail} failed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()