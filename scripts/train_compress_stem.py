#!/usr/bin/env python3
"""调用 runtime.train_compress_stem：DepthCompress stem 训练。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.train_compress_stem import main

if __name__ == "__main__":
    main()
