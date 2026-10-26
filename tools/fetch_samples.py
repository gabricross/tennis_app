from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HOME
from media import probe, snapshot
from setup_models import download, sha256

SAMPLES = {
    "davis03": "File:2018 Davis Cup Americas Zone - Uruguay vs Mexico - 03.webm",
    "davis06": "File:2018 Davis Cup Americas Zone - Uruguay vs Mexico - 06.webm",
    "rola": "File:Blaž Rola NCAAs 2013.webm",
}


def main() -> None:
    root = HOME / "fixtures"
    root.mkdir(exist_ok=True)
    manifest = {}
    for name, title in SAMPLES.items():
        query = urllib.parse.urlencode(
            {"action": "query", "titles": title, "prop": "imageinfo",
             "iiprop": "url|size|extmetadata", "format": "json"}
        )
        request = urllib.request.Request(
            f"https://commons.wikimedia.org/w/api.php?{query}",
            headers={"User-Agent": "TennisLab-local-test/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        info = next(iter(data["query"]["pages"].values()))["imageinfo"][0]
        meta = info["extmetadata"]
        if meta["LicenseShortName"]["value"] not in ("CC BY-SA 4.0", "CC BY 3.0"):
            raise ValueError(f"Unexpected sample license: {title}")
        directory = root / name
        directory.mkdir(exist_ok=True)
        video = directory / "sample.webm"
        download(info["url"], video)
        metadata = probe(video)
        for fraction in (0.1, 0.5, 0.8):
            snapshot(video, metadata["duration"] * fraction, directory / f"frame_{fraction}.jpg")
        manifest[name] = {
            "path": str(video), "title": title.removeprefix("File:"),
            "author": html.unescape(re.sub("<[^>]+>", "", meta["Artist"]["value"])).strip(),
            "license": meta["LicenseShortName"]["value"],
            "license_url": meta["LicenseUrl"]["value"], "source": info["descriptionurl"],
            "sha256": sha256(video), "metadata": metadata,
        }
        print(name, json.dumps(metadata), flush=True)
    (root / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
