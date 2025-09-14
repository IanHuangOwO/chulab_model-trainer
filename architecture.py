"""
Model architecture inspector for saved PyTorch models.

This utility loads a model artifact (either a pickled `torch.nn.Module`, a
TorchScript file, or a checkpoint/state_dict) and prints a readable summary to
stdout. It is useful for quickly checking the network structure and parameter
counts without spinning up training or inference code.

What it does
- If given a TorchScript file (or `--jit`), loads via `torch.jit.load` and
  prints the scripted module and, when available, the inlined graph.
- If given a pickled `nn.Module` (common .pth/.pt), uses `torch.load` and
  prints the module architecture plus total/trainable parameter counts.
- If given a checkpoint/state_dict, lists the state_dict keys (and tensor
  shapes) to help you reconstruct the original model for inspection.

Common usage
  python architecture.py ./datas/weights/model.pth --project-root .
  python architecture.py ./datas/weights/model.ts --jit

Exit codes
- 0: Successful load and summary printed.
- 1: Could not load or unsupported object type.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def print_module_info(module: torch.nn.Module):
    """Print a concise architecture summary for a `torch.nn.Module`.

    Shows the module's class, the textual architecture representation, and
    parameter counts (total and trainable).

    Parameters
    - module: Instantiated PyTorch module to summarize.
    """
    print("Model type:", type(module))
    print("\nArchitecture:\n")
    print(module)
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"\nParameters: total={total:,}, trainable={trainable:,}")


def main():
    """CLI entrypoint.

    Parses arguments, attempts to load the provided model artifact, and prints
    a summary appropriate to the loaded object type. Returns an exit code
    compatible with shell usage (0 success, 1 failure).
    """
    parser = argparse.ArgumentParser(description="Print architecture from a .pth/.pt model file.")
    parser.add_argument("path", help="Path to model file (.pth/.pt)")
    parser.add_argument("--jit", action="store_true", help="Force TorchScript loader (torch.jit.load)")
    parser.add_argument("--map-location", default="cpu", help="Device for loading (default: cpu)")
    parser.add_argument("--project-root", help="Path to project root to make packages importable (adds to sys.path)")
    args = parser.parse_args()

    model_path = Path(args.path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    # Ensure project root (containing 'models', etc.) is importable for torch.load of pickled nn.Module
    if args.project_root:
        root = Path(args.project_root).resolve()
    else:
        # Default: repo root = parent of utils/
        root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # 1) Try TorchScript first (unless user forces otherwise via not using --jit?)
    if args.jit:
        try:
            ts = torch.jit.load(str(model_path), map_location=args.map_location)
            print("Loaded TorchScript model.")
            print("\nTorchScript Module:\n")
            print(ts)
            try:
                print("\nGraph:\n")
                print(ts.inlined_graph)
            except Exception:
                pass
            return 0
        except Exception as e:
            print(f"TorchScript load failed: {e}")
            return 1

    # 2) Try standard torch.load (pickle). This requires the original class to be importable
    try:
        obj = torch.load(str(model_path), map_location=args.map_location, weights_only=False)
    except Exception as e:
        # Maybe it is TorchScript but user didn't pass --jit
        try:
            ts = torch.jit.load(str(model_path), map_location=args.map_location)
            print("Loaded TorchScript model.")
            print("\nTorchScript Module:\n")
            print(ts)
            try:
                print("\nGraph:\n")
                print(ts.inlined_graph)
            except Exception:
                pass
            return 0
        except Exception as e2:
            print(f"torch.load failed: {e}")
            print(f"torch.jit.load failed: {e2}")
            return 1

    # 3) If it's a module, print architecture and parameter counts
    if isinstance(obj, torch.nn.Module):
        print_module_info(obj)
        return 0

    # 4) If it's a dict (state_dict or checkpoint), summarize keys
    if isinstance(obj, dict):
        print("Loaded checkpoint/state_dict dict.\n")
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        else:
            sd = obj

        keys = list(sd.keys())
        print(f"Keys: {len(keys)} parameters")
        preview = 20
        for k in keys[:preview]:
            try:
                shape = tuple(sd[k].shape)
            except Exception:
                shape = "?"
            print(f"  {k}: {shape}")
        if len(keys) > preview:
            print(f"  ... (+{len(keys) - preview} more)")
        print("\nNote: This is a state_dict; to print the full architecture, instantiate the model class and load the state_dict.")
        return 0

    print(f"Loaded object of type {type(obj)}; not a Module or dict. Cannot print architecture.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
