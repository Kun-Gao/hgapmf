from __future__ import annotations

import argparse

import torch

from hgapmf import Config, OUTPUT_CHANNELS, HGAPMFNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal HG-APMF forward smoke test.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--prototype-dim", type=int, default=16)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 64, 64), metavar=("D", "H", "W"))
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = HGAPMFNet(
        in_channels=args.channels,
        num_classes=15,
        base_channels=args.base_channels,
        config=Config(prototype_dim=args.prototype_dim),
    ).to(device)
    model.eval()

    x = torch.randn(args.batch_size, args.channels, *args.shape, device=device)
    with torch.no_grad():
        for mode in ("ct_mr", "ct_only", "mr_only"):
            output = model(x, return_debug=True, mode=mode)
            progressive = output["progressive"]
            shapes = [tuple(t.shape) for t in progressive]
            channels = [t.shape[1] for t in progressive]
            if channels != list(OUTPUT_CHANNELS):
                raise RuntimeError(f"Unexpected output channels for {mode}: {channels}")
            if not torch.isfinite(output["seg"]).all():
                raise RuntimeError(f"Non-finite final logits for {mode}")
            print(f"{mode}: final={tuple(output['seg'].shape)} progressive={shapes}")

    print("HG-APMF smoke test passed.")


if __name__ == "__main__":
    main()
