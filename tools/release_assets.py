"""Pack a manifest's reference products into release assets and record each one's URL and sha256 in the manifest.

    python tools/release_assets.py MANIFEST --store DIR --out DIR --url https://github.com/OWNER/REPO/releases/download/TAG

Each product ``<id>`` of MANIFEST becomes ``OUT/<id>.tar.gz``: its files (the manifest's list, each checked against its
sha256 first) at the top of the archive, written byte for byte reproducibly (sorted, no times or owners). The manifest
gains ``"url": "<URL>/<id>.tar.gz"`` and ``"sha256"`` (the archive's) on each product, which is what
``lumen.compare.references.product`` fetches. Upload every file of OUT to the release named by URL.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def pack(src: Path, files: dict, out: Path) -> None:
    """Write ``out`` (.tar.gz) from the listed files of directory ``src``, reproducibly."""
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(files):
                if sha256(src / name) != files[name]:
                    raise SystemExit(f"{src / name}: sha256 differs from the manifest ({files[name]})")
                data = (src / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--store", type=Path, required=True, help="the directory holding <id>/ of every product")
    ap.add_argument("--out", type=Path, required=True, help="where the assets are written")
    ap.add_argument("--url", required=True, help="the release's download URL the assets are uploaded to")
    a = ap.parse_args(argv)
    m = json.loads(a.manifest.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for pid, entry in m["products"].items():
        asset = a.out / f"{pid}.tar.gz"
        pack(a.store / pid, entry["files"], asset)
        entry["url"], entry["sha256"] = f"{a.url.rstrip('/')}/{asset.name}", sha256(asset)
        print(f"{asset.name}: {asset.stat().st_size / 1e6:.1f} MB")
    a.manifest.write_text(json.dumps(m, indent=1) + "\n")


if __name__ == "__main__":
    main()
