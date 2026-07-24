"""Shared insert-only logic for PegInsertionVertical data collection and conversion.

This module is the single source of truth for two things used by both
``collect_peg_insertion_controller_data.py`` (collection-time ``--collect-mode
insert_only``) and ``convert_to_insert_only.py`` (post-hoc conversion of a full
dataset):

1. ``find_lift_end`` -- the crop criterion that locates the lift-end frame where
   the move-and-insert segment begins (drop the reach/grasp/close/lift prefix).
2. ``generate_insert_only_prompts`` -- single-stage "insert the peg into the
   hole" prompt variants (two-stage "pick up ... and ..." wording excluded).

Keeping these here guarantees the collection-time insert-only output and the
post-hoc conversion produce byte-identical crops and prompt sets for the same
seed. The prompt generator and its ``_imperative`` are copied verbatim from the
original ``convert_to_insert_only.py`` so existing converted datasets remain
reproducible.
"""

from __future__ import annotations

import random

import numpy as np

EVAL_PROMPT = "insert the peg into the hole"

# Insert-only prompt vocabulary (single-stage only; no "pick up ... and ...").
_VERBS = ["insert", "place", "put", "drop", "fit", "slide", "push", "lower", "set"]
_PEG = ["peg", "peg", "peg", "small peg", "peg piece", "block peg", "peg block"]
_DIR = ["vertically", "straight down", "top-down", "downward", "straight down into"]
_TARGET = ["hole", "hole", "hole", "box hole", "target hole", "slot", "opening", "recess"]
_PEG_COLOR = ["blue", "blue", "blue", "", ""]
_HOLE_COLOR = ["orange", "orange", "orange", "", ""]
_INTO = ["into", "into", "into", "in", "inside"]


def _make_target(color: str) -> str:
    return f"the {color} hole" if color else "the hole"


def _make_peg_phrase(color: str, peg: str) -> str:
    if color:
        return f"the {color} peg"
    if peg == "peg":
        return "the peg"
    return f"the {peg}"


def _imperative(rng: random.Random) -> str:
    verb = rng.choice(_VERBS)
    peg_phrase = _make_peg_phrase(rng.choice(_PEG_COLOR), rng.choice(_PEG))
    target = _make_target(rng.choice(_HOLE_COLOR))
    prep = rng.choice(_INTO)
    if rng.random() < 0.5:
        direction = rng.choice(_DIR)
        if direction == "straight down into":
            return f"{verb} {peg_phrase} straight down {prep} {target}"
        return f"{verb} {peg_phrase} {direction} {prep} {target}"
    return f"{verb} {peg_phrase} {prep} {target}"


def generate_insert_only_prompts(num: int, seed: int) -> list[str]:
    """Single-stage insert-only prompts; index 0 fixed to EVAL_PROMPT."""
    if num < 1:
        raise ValueError("num must be positive")
    rng = random.Random(seed)
    prompts = [EVAL_PROMPT]
    seen = {EVAL_PROMPT}
    attempts = 0
    while len(prompts) < num and attempts < num * 200:
        attempts += 1
        prompt = _imperative(rng)
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    if len(prompts) < num:
        raise RuntimeError(f"could only generate {len(prompts)} unique prompts (wanted {num})")
    return prompts[:num]


# --- crop criterion ---------------------------------------------------------

LIFT_Z_GAIN = 0.05      # z must rise this much above z[t_close] to confirm lift
PLATEAU_EPS_Z = 0.003   # |dz/dt| below this => vertical plateau
PLATEAU_EPS_XY = 0.005  # |dxy/dt| below this => no lateral motion yet
SEARCH_WINDOW = 60      # frames after t_close to search for lift plateau


def find_lift_end(actions: np.ndarray, state_tcp: np.ndarray) -> int | None:
    """Return the lift-end frame index, or None if no reliable boundary."""
    g = actions[:, 6]
    flips = np.where(np.diff(np.sign(g)) != 0)[0]
    if len(flips) == 0:
        return None
    t_close = int(flips[0]) + 1
    z = state_tcp[:, 2]
    xy = state_tcp[:, :2]
    dz = np.abs(np.diff(z))
    dxy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if z[t_close] + LIFT_Z_GAIN >= z.max():
        return None
    upper = min(t_close + SEARCH_WINDOW, len(z) - 2)
    for i in range(t_close, upper):
        lo = max(i - 1, 0)
        hi = i + 2
        if dz[lo:hi].mean() < PLATEAU_EPS_Z and dxy[lo:hi].mean() < PLATEAU_EPS_XY \
                and z[i] > z[t_close] + LIFT_Z_GAIN:
            return i
    return None
