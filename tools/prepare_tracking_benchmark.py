from __future__ import annotations

import html
import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HOME
from media import probe, snapshot
from setup_models import download, sha256

LOGGER = logging.getLogger("tennis.benchmark")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    root = HOME / "tracking_benchmark"
    root.mkdir(exist_ok=True)
    sources = []
    titles = [
        (f"davis{number:02}", f"2018 Davis Cup Americas Zone - Uruguay vs Mexico - {number:02}.webm")
        for number in (3, 6)
    ] + [
        ("rola", "Blaž Rola NCAAs 2013.webm"),
    ]
    for name, title in titles:
        query = urllib.parse.urlencode({
            "action": "query", "titles": "File:" + title, "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "format": "json",
        })
        request = urllib.request.Request(
            "https://commons.wikimedia.org/w/api.php?" + query,
            headers={"User-Agent": "TennisLab-local-test/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            info = next(iter(json.load(response)["query"]["pages"].values()))["imageinfo"][0]
        meta = info["extmetadata"]
        license_name = meta["LicenseShortName"]["value"]
        if license_name not in ("CC BY-SA 4.0", "CC BY 3.0", "CC BY-SA 2.5 co"):
            raise ValueError(f"Unreviewed source license: {title}: {license_name}")
        directory = root / name
        directory.mkdir(exist_ok=True)
        existing = HOME / "fixtures" / name / "sample.webm"
        original = existing if existing.exists() else directory / ("source" + Path(title).suffix)
        download(info["url"], original)
        video = original
        sources.append({
            "id": name, "path": str(video), "original_sha256": sha256(original),
            "source_credit": {
                "author": html.unescape(re.sub("<[^>]+>", "", meta["Artist"]["value"])).strip(),
                "provider": "Wikimedia Commons", "license": license_name,
                "license_url": meta["LicenseUrl"]["value"], "source_url": info["descriptionurl"],
            },
            "metadata": probe(video),
        })
    pexels = [
        ("young", "young-men-playing-tennis-5740607", "5740607/5740607-uhd_2732_1440_25fps.mp4", "cottonbro studio"),
        ("rally", "people-playing-tennis-992693", "992693/992693-hd_1920_1080_25fps.mp4", "jessica politi"),
        ("aerial", "aerial-view-of-tennis-players-on-court-35634600", "35634600/15102041_1080_1920_60fps.mp4", "Cansu Hangül"),
        ("girls", "girls-playing-tennis-8224226", "8224226/8224226-hd_1920_1080_30fps.mp4", "RDNE Stock project"),
        ("clay", "aerial-view-of-tennis-match-on-clay-court-34963195", "34963195/14811213_1920_1080_30fps.mp4", "TC Herzele Laddertornooi"),
        ("audience", "men-playing-tennis-with-audience-10390967", "10390967/10390967-hd_1920_1080_30fps.mp4", "Gelato Prod"),
    ]
    for name, slug, media, author in pexels:
        directory = root / name
        directory.mkdir(exist_ok=True)
        video = directory / "source.mp4"
        download("https://videos.pexels.com/video-files/" + media, video)
        sources.append({
            "id": name, "path": str(video), "original_sha256": sha256(video),
            "source_credit": {
                "author": author, "provider": "Pexels", "license": "Pexels License",
                "license_url": "https://www.pexels.com/license/",
                "source_url": "https://www.pexels.com/video/" + slug + "/",
            }, "metadata": probe(video),
        })
    night = HOME / "real_demo" / "night_match.mp4"
    metadata = json.loads((HOME / "real_demo" / "night_source.json").read_text(encoding="utf-8"))
    sources.append({
        "id": "night", "path": str(night), "original_sha256": sha256(night),
        "source_credit": {
            "author": metadata["creator"]["name"], "provider": "Pexels", "license": "Pexels License",
            "license_url": metadata["license"], "source_url": metadata["url"],
        }, "metadata": probe(night),
    })
    (root / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    for source in sources:
        directory = root / source["id"]
        directory.mkdir(exist_ok=True)
        images = []
        for requested in (0, 1, 2, 3, 5, 8):
            second = min(requested, source["metadata"]["duration"] - 0.2)
            frame_path = directory / f"review_{second}.jpg"
            snapshot(Path(source["path"]), second, frame_path)
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"Cannot read review frame for {source['id']}")
            image = cv2.resize(image, (640, 360))
            cv2.putText(image, f"{source['id']} | {second}s", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            images.append(image)
        sheet = np.vstack([np.hstack(images[:3]), np.hstack(images[3:])])
        if not cv2.imwrite(str(directory / "review.jpg"), sheet):
            raise RuntimeError("Cannot write review sheet")
    LOGGER.info("Prepared %d distinct source files in %s", len(sources), root)


if __name__ == "__main__":
    main()
