"""Logging and Experiment Tracking Utilities."""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional


def setup_logger(output_dir: str, name: str = "VideoDiffusion") -> logging.Logger:
    """Configures structured file and console logger."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class MetricTracker:
    """Tracks running averages of training losses and throughput."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}
        self.start_time = time.time()

    def update(self, key: str, val: float, n: int = 1):
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0
        self.metrics[key] += val * n
        self.counts[key] += n

    def get_average(self, key: str) -> float:
        if key not in self.metrics or self.counts[key] == 0:
            return 0.0
        return self.metrics[key] / self.counts[key]

    def summary(self) -> Dict[str, float]:
        return {k: self.get_average(k) for k in self.metrics}

