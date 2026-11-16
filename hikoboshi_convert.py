import argparse
from pathlib import Path

import torch

from configs import config

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = PROJECT_ROOT / "weights" / "Marin.pth"
INDEX = PROJECT_ROOT / "logs" / "added_IVF1116_Flat_nprobe_1_Marin_v2.index"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config.infer_device = torch.device("cpu")
    config.infer_dtype = torch.float32
    config.DML_AVAILABLE = False
    config.DML_DEVICE = None
    torch.set_num_threads(8)

    from infer.cli import main as infer_main

    return infer_main(
        [
            "--model",
            str(MODEL),
            "--index",
            str(INDEX),
            "--input",
            args.input,
            "--output",
            args.output,
            "--f0-method",
            "pm",
            "--index-rate",
            "0.75",
            "--overwrite",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
