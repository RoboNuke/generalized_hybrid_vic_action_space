#!/usr/bin/env python3
"""Split requirements.lock.txt into three install groups for the HPC image build.

The Isaac Sim / torch wheels live on dedicated indexes, so they must be pip-installed
separately from the PyPI tail. This reads the lock file and writes three plain
requirements files (one package per line, comments/blanks stripped):

    <out>/torch.txt      -> install with  --index-url https://download.pytorch.org/whl/cu128
    <out>/isaacsim.txt   -> install with  --extra-index-url https://pypi.nvidia.com
    <out>/rest.txt       -> install from   PyPI (+ any git URLs in the lock)

Usage:  split_requirements.py <requirements.lock.txt> <out_dir>
"""
import sys
import os


def name(spec: str) -> str:
    # package name from "pkg==x", "pkg @ git+...", etc. -> normalized lowercase
    return spec.split("==")[0].split(" @")[0].strip().lower().replace("_", "-")


def main() -> int:
    lock, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch, isaacsim, rest = [], [], []
    for line in open(lock):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n = name(line)
        if n in ("torch", "torchvision", "torchaudio"):
            torch.append(line)
        elif n.startswith("isaacsim"):
            isaacsim.append(line)
        elif n.startswith("isaaclab"):
            continue  # installed from IsaacLab source, never from an index
        else:
            rest.append(line)
    for fname, pkgs in (("torch", torch), ("isaacsim", isaacsim), ("rest", rest)):
        with open(os.path.join(out, f"{fname}.txt"), "w") as f:
            f.write("\n".join(pkgs) + "\n")
    print(f"split: torch={len(torch)} isaacsim={len(isaacsim)} rest={len(rest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
