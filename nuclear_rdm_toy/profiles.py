from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    width: int
    rank: int
    point_depth: int
    pair_depth: int
    rff_features: int


PROFILES = {
    "baseline": ModelProfile(192, 16, 3, 2, 32),
    "medium": ModelProfile(96, 4, 2, 2, 16),
    "small": ModelProfile(64, 2, 2, 1, 16),
    "tiny": ModelProfile(32, 1, 1, 1, 8),
}

PARTICLE_MASSES = {
    "electron": 1.0,
    "m10": 10.0,
    "m100": 100.0,
    "muon": 206.768283,
    "proton": 1836.152673,
}
