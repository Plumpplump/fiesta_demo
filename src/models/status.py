"""Turn tracks into the rows the status panel shows (pure, testable)."""

from __future__ import annotations

import time

from . import constants as C
from .types import CHECKING, Track


def needs_classification(track: Track, now: float) -> bool:
    """True when this person has no verdict yet, or the verdict is stale."""
    if track.uniform == CHECKING:
        return True
    return (now - track.last_classified) >= C.RECLASSIFY_INTERVAL


def build_status_rows(tracks: list[Track]) -> list[dict]:
    """One row per visible person, numbered left-to-right, never by name."""
    ordered = sorted(tracks, key=lambda t: (t.label_index, t.box[0]))
    rows: list[dict] = []
    for track in ordered:
        rows.append(
            {
                "label": f"{C.PERSON_LABEL_PREFIX}{track.label_index}",
                "track_id": track.track_id,
                "uniform": track.uniform,
                "uniform_score": track.uniform_score,
                "glasses": track.glasses,
                "glasses_score": track.glasses_score,
                "detail": track.uniform_detail,
            }
        )
    return rows


def frame_timestamp() -> float:
    return time.monotonic()