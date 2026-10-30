"""Export locally supplied WASB tennis weights (Tarashima et al., BMVC 2023) to ONNX.

The WASB-SBDT code is MIT-licensed, but the licence of its pretrained tennis
checkpoint is not stated upstream (nttcom/WASB-SBDT issue #12). This tool never
downloads or redistributes weights: the user supplies both the cloned repository
and the checkpoint, and the exported model stays in the local model folder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS


def _stage(branches: int, block: str, blocks: list[int], channels: list[int]) -> dict:
    return {"NUM_MODULES": 1, "NUM_BRANCHES": branches, "BLOCK": block, "NUM_BLOCKS": blocks,
            "NUM_CHANNELS": channels, "FUSE_METHOD": "SUM"}


# Transcribed from WASB-SBDT src/configs/model/wasb.yaml.
CONFIG = {
    "name": "hrnet", "frames_in": 3, "frames_out": 3, "inp_height": 288, "inp_width": 512,
    "out_height": 288, "out_width": 512, "rgb_diff": False, "out_scales": [0],
    "MODEL": {"INIT_WEIGHTS": True, "EXTRA": {
        "FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ["*"], "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
        "STAGE1": _stage(1, "BOTTLENECK", [1], [32]),
        "STAGE2": _stage(2, "BASIC", [2, 2], [16, 32]),
        "STAGE3": _stage(3, "BASIC", [2, 2, 2], [16, 32, 64]),
        "STAGE4": _stage(4, "BASIC", [2, 2, 2, 2], [16, 32, 64, 128]),
        "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
    }},
}


class _Config(dict):
    """Attribute access, as the upstream OmegaConf configuration provides."""

    def __getattr__(self, key: str):
        return self[key]


def _wrap(value):
    return _Config({key: _wrap(item) for key, item in value.items()}) if isinstance(value, dict) else value


def main() -> None:
    import onnxruntime
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Local clone of nttcom/WASB-SBDT")
    parser.add_argument("--weights", type=Path, required=True, help="Local wasb_tennis_best.pth.tar")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("wasb_hrnet", args.repo / "src" / "models" / "hrnet.py")
    hrnet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hrnet)
    model = hrnet.HRNet(_wrap(CONFIG))
    # weights_only refuses arbitrary pickled code in a third-party checkpoint.
    state = torch.load(args.weights, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    model.eval()

    class Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, frames):
            return self.model(frames)[0]

    target = MODELS / "ball_wasb.onnx"
    sample = torch.randn(1, 9, 288, 512)
    # Reference first: the legacy exporter leaves the PyTorch module altered afterwards.
    with torch.no_grad():
        expected = Logits()(sample).numpy()
    torch.onnx.export(Logits(), sample, str(target), input_names=["frames"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    exported = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    error = float(np.abs(exported.run(None, {"frames": sample.numpy()})[0] - expected).max())
    if error > 1e-3:
        target.unlink()
        raise RuntimeError(f"ONNX export differs from PyTorch by {error:.2e}.")
    cards = MODELS / "model_cards"
    cards.mkdir(exist_ok=True)
    (cards / "ball_wasb.json").write_text(json.dumps({
        "model": "WASB HRNet, tennis checkpoint",
        "paper": "Tarashima et al., Widely Applicable Strong Baseline for Sports Ball Detection and Tracking, BMVC 2023",
        "code": "https://github.com/nttcom/WASB-SBDT (MIT)",
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "max_abs_logit_difference": error,
        "license": "Checkpoint licence not stated upstream; local evaluation only, do not redistribute.",
    }, indent=2), encoding="utf-8")
    print(f"Exported {target} (max logit difference {error:.2e}).")


if __name__ == "__main__":
    main()
