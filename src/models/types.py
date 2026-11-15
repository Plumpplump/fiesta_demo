# Author: David Li, SISHK
"""Dataclasses shared by the detection pipeline (no Flet imports)."""

from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]

# Label vocabularies. Strings (not enums) keep the dataclasses trivially
# printable/serialisable in tests and in the UI.
IN_UNIFORM = "in uniform"
NOT_IN_UNIFORM = "not in uniform"
UNSURE = "unsure"
CHECKING = "checking"

GLASSES = "glasses"
NO_GLASSES = "no glasses"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Detection:
    """One detected person, box in original-frame pixel coordinates."""

    box: Box
    score: float


@dataclass
class Track:
    """A person followed across frames."""

    track_id: int
    label_index: int
    box: Box
    torso_box: Box
    hits: int = 1
    misses: int = 0
    # A track only becomes visible/judgeable after two hits: the first box of
    # somebody stepping into view is usually partial (measured).
    confirmed: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_classified: float = 0.0
    classified_box: Box | None = None
    uniform: str = CHECKING
    uniform_score: float | None = None
    uniform_detail: str = ""
    glasses: str = UNKNOWN
    glasses_score: float | None = None
    glasses_detail: str = ""
    # Every check appends its probability; the card publishes a verdict from
    # the median of the last few samples (see status.smoothed_score).
    uniform_scores: list[float] = field(default_factory=list)
    glasses_scores: list[float] = field(default_factory=list)
    white_ratio: float = 0.0
    red_ratio: float = 0.0
    checks: int = 0  # classification passes taken (drives the warm-up gate)
    eye_box: Box | None = None
    face_box: Box | None = None
    notes: list[str] = field(default_factory=list)