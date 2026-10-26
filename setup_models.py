from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import urllib.request
from pathlib import Path

from config import MODELS

LOGGER = logging.getLogger("models")
ASSETS = {
    "court": {
        "repo": "kjfk/tennis-court-keypoints-mirror",
        "revision": "68342e926f5d012c79a02f8cc045bb75ca443ec8",
        "file": "keypoints_model.pth",
        "sha256": "16ebb7e46dc88247440c86b388e4f07f0d4abb76ce0a01a22925d3163f7fb7f3",
        "license_declared_in_model_card": "MIT",
        "origin": "abdullahtarek/tennis_analysis",
    },
    "ball": {
        "repo": "kjfk/tracknet-tennis-ball-mirror",
        "revision": "1cd650376c405fed6ae5103bad704310a6e52dda",
        "file": "tracknet_tennis.pt",
        "sha256": "c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076",
        "license_declared_in_model_card": "MIT",
        "origin": "yastrebksv/TrackNet",
    },
}
PEOPLE_SHA = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
PEOPLE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/"
    "object_detection_yolox_2022nov.onnx"
)
PEOPLE_LICENSE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/"
    "48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/LICENSE"
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, target: Path, expected: str | None = None) -> None:
    if target.exists():
        if expected is not None and sha256(target) != expected:
            raise ValueError(f"Existing model failed SHA-256 verification: {target.name}")
        LOGGER.info("Verified existing %s", target.name)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "TennisLab/1.0 (local inference)"})
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if expected is not None and sha256(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"Download failed SHA-256 verification: {target.name}")
    temporary.replace(target)
    LOGGER.info("Downloaded %s (%d bytes)", target.name, target.stat().st_size)


def convert_court() -> None:
    import torch
    from torchvision.models import resnet50

    state = torch.load(MODELS / ASSETS["court"]["file"], map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 28)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model, torch.zeros(1, 3, 224, 224), str(MODELS / "court.onnx"),
            input_names=["image"], output_names=["points"], opset_version=17, dynamo=False,
        )
    LOGGER.info("Exported court.onnx")


def convert_ball() -> None:
    import torch
    from torch import nn

    state = torch.load(MODELS / ASSETS["ball"]["file"], map_location="cpu", weights_only=True)
    stages = []
    for index in range(1, 19):
        prefix = f"conv{index}.block."
        kernel = state[prefix + "0.weight"]
        outputs, inputs = kernel.shape[:2]
        block = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(outputs),
        )
        block.load_state_dict(
            {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)},
            strict=True,
        )
        stages.append(block)
        if index in (2, 4, 7):
            stages.append(nn.MaxPool2d(2))
        if index in (10, 13, 15):
            stages.append(nn.Upsample(scale_factor=2, mode="nearest"))

    class IntensityMap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(*stages)

        def forward(self, images):
            return self.network(images).argmax(dim=1).to(torch.uint8)

    model = IntensityMap().eval()
    torch.set_num_threads(2)
    with torch.inference_mode():
        torch.onnx.export(
            model, torch.zeros(1, 9, 360, 640), str(MODELS / "ball.onnx"),
            input_names=["images"], output_names=["intensity"],
            opset_version=17, dynamo=False,
        )
    LOGGER.info("Exported ball.onnx; the output is intensity classes, not probability")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare verified local TennisLab models.")
    parser.add_argument("--convert", choices=("court", "ball"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--players-only", action="store_true", help="Download only the Apache-2.0 person model.")
    arguments = parser.parse_args()
    if arguments.convert and arguments.players_only:
        parser.error("--players-only cannot be combined with --convert.")
    if arguments.convert:
        (convert_court if arguments.convert == "court" else convert_ball)()
        return
    people = MODELS / "people.onnx"
    download(PEOPLE_URL, people, PEOPLE_SHA)
    cards = MODELS / "model_cards"
    cards.mkdir(exist_ok=True)
    download(PEOPLE_LICENSE_URL, cards / "people-Apache-2.0.txt")
    selected_assets = {} if arguments.players_only else ASSETS
    for name, asset in selected_assets.items():
        base = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/"
        download(base + asset["file"], MODELS / asset["file"], asset["sha256"])
        download(base + "README.md", cards / f"{name}.md")
    manifest = dict(selected_assets)
    manifest["people"] = {"source": PEOPLE_URL, "sha256": PEOPLE_SHA, "license": "Apache-2.0"}
    manifest["notice"] = (
        "Third-party pretrained weights. Model-card licences are declared by their uploaders; "
        "the original court and TrackNet source repositories do not provide a LICENSE. "
        "No upstream source code is redistributed. Do not publish these mirrored assets "
        "as your own models; review provenance before commercial redistribution."
    )
    (MODELS / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not arguments.download_only and not arguments.players_only:
        raise SystemExit(
            "Downloads verified. Run --convert court and --convert ball in separate processes "
            "to keep export memory bounded."
        )


if __name__ == "__main__":
    main()
