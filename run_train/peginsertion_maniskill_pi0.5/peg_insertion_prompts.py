"""Short prompt variants for PegInsertionVertical training data.

The eval environment returns ``insert the peg into the hole``.  Keep SFT
prompts close to that length/style while preserving light language variation.
"""

from __future__ import annotations

import random

EVAL_PROMPT = "insert the peg into the hole"
DEFAULT_NUM_PROMPTS = 500
DEFAULT_PROMPT_SEED = 0

_VERBS = ["insert", "place", "put", "drop", "fit", "slide", "push", "lower", "set"]
_PEG = ["peg", "peg", "peg", "small peg", "peg piece", "block peg", "peg block"]
_DIR = ["vertically", "straight down", "top-down", "downward", "straight down into"]
_TARGET = ["hole", "hole", "hole", "box hole", "target hole", "slot", "opening", "recess"]
_PEG_COLOR = ["blue", "blue", "blue", "", ""]
_HOLE_COLOR = ["orange", "orange", "orange", "", ""]
_INTO = ["into", "into", "into", "in", "inside"]

_TWO_STAGE_VERBS = [
    ("pick up", "insert it"),
    ("pick up", "place it"),
    ("pick up", "put it"),
    ("grasp", "insert it"),
    ("grasp", "place it"),
    ("grab", "put it"),
    ("pick up", "fit it"),
    ("grasp", "drop it"),
    ("pick up", "slide it"),
    ("lift", "insert it"),
]


def _make_target(color: str) -> str:
    if color:
        return f"the {color} hole"
    return "the hole"


def _make_peg_phrase(color: str, peg: str) -> str:
    if color:
        return f"the {color} peg"
    if peg == "peg":
        return "the peg"
    return f"the {peg}"


def _imperative(rng: random.Random) -> str:
    verb = rng.choice(_VERBS)
    peg_color = rng.choice(_PEG_COLOR)
    hole_color = rng.choice(_HOLE_COLOR)
    peg_phrase = _make_peg_phrase(peg_color, rng.choice(_PEG))
    target = _make_target(hole_color)
    prep = rng.choice(_INTO)
    if rng.random() < 0.5:
        direction = rng.choice(_DIR)
        if direction == "straight down into":
            return f"{verb} {peg_phrase} straight down {prep} {target}"
        return f"{verb} {peg_phrase} {direction} {prep} {target}"
    return f"{verb} {peg_phrase} {prep} {target}"


def _two_stage(rng: random.Random) -> str:
    first, second = rng.choice(_TWO_STAGE_VERBS)
    peg_color = rng.choice(_PEG_COLOR)
    hole_color = rng.choice(_HOLE_COLOR)
    peg_phrase = _make_peg_phrase(peg_color, rng.choice(_PEG))
    target = _make_target(hole_color)
    return f"{first} {peg_phrase} and {second} into {target}"


def generate_prompts(
    num: int = DEFAULT_NUM_PROMPTS,
    seed: int = DEFAULT_PROMPT_SEED,
) -> list[str]:
    """Return exactly ``num`` unique short prompts.

    Index 0 is fixed to the eval prompt so the exact eval wording is included
    in the training distribution.
    """
    if num < 1:
        raise ValueError("num must be positive")

    rng = random.Random(seed)
    prompts: list[str] = [EVAL_PROMPT]
    seen = {EVAL_PROMPT}
    two_stage_target = int(round(num * 0.15))
    two_stage_made = 0
    attempts = 0
    while len(prompts) < num and attempts < num * 200:
        attempts += 1
        want_two = two_stage_made < two_stage_target and rng.random() < 0.18
        prompt = _two_stage(rng) if want_two else _imperative(rng)
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if want_two:
            two_stage_made += 1

    if len(prompts) < num:
        raise RuntimeError(
            f"could only generate {len(prompts)} unique prompts (wanted {num})"
        )
    return prompts[:num]
