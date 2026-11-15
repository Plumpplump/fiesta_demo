# Author: David Li, SISHK
"""Turn tracks into the rows the status panel shows (pure, testable)."""

from __future__ import annotations

import math
import statistics
import time

from . import constants as C
from .types import CHECKING, Track


def moved_a_lot(track: Track) -> bool:
    """True when the person moved or changed size enough to invalidate the crop.

    Both checks use 20% of the box height, like the sibling project: a step to
    the side changes the crop, and so does walking toward the camera (the box
    grows, and the shirt occupies a different fraction of it).
    """
    if track.classified_box is None:
        return False
    cx0 = (track.classified_box[0] + track.classified_box[2]) / 2.0
    cy0 = (track.classified_box[1] + track.classified_box[3]) / 2.0
    cx1 = (track.box[0] + track.box[2]) / 2.0
    cy1 = (track.box[1] + track.box[3]) / 2.0
    height = max(1.0, track.box[3] - track.box[1])
    previous_height = max(1.0, track.classified_box[3] - track.classified_box[1])
    shift = math.hypot(cx1 - cx0, cy1 - cy0)
    resized = abs(height - previous_height)
    return (
        shift >= C.RECLASSIFY_MOVE_RATIO * height
        or resized >= C.RECLASSIFY_MOVE_RATIO * previous_height
    )


def smoothed_score(scores: list[float], window: int = C.SMOOTHING_WINDOW) -> float | None:
    """Median of the most recent samples, or ``None`` before warm-up.

    The crop shifts slightly between checks and the reference cosine moves
    +/-0.03, which is enough to swing a steep sigmoid from 0.15 to 0.59. A
    median over a few samples keeps the published verdict stable instead of
    flipping (or holding "checking..." forever when two samples disagree).
    """
    if len(scores) < C.WARMUP_SAMPLES:
        return None
    recent = scores[-window:]
    return float(statistics.median(recent))


def needs_classification(track: Track, now: float) -> bool:
    """True when this person needs a (first or refresh) classification.

    Nobody is judged before the track settles, and the first two checks are
    taken close together so a stable verdict appears quickly.
    """
    if not track.confirmed:
        return False
    if (now - track.first_seen) < C.TRACK_SETTLE_SECONDS:
        return False
    if track.checks < C.WARMUP_SAMPLES:
        return (now - track.last_classified) >= C.WARMUP_INTERVAL
    if track.last_classified <= 0.0:
        return True
    if (now - track.last_classified) >= C.RECLASSIFY_INTERVAL:
        return True
    return moved_a_lot(track)


def build_status_rows(tracks: list[Track]) -> list[dict]:
    """One row per visible person, numbered left-to-right, never by name."""
    ordered = sorted(tracks, key=lambda t: (t.label_index, t.box[0]))
    rows: list[dict] = []
    for track in ordered:
        details = [d for d in (track.uniform_detail, track.glasses_detail) if d]
        rows.append(
            {
                "label": f"{C.PERSON_LABEL_PREFIX}{track.label_index}",
                "track_id": track.track_id,
                "uniform": track.uniform,
                # No score next to "checking...": it would show the raw number
                # of a verdict that is not published yet.
                "uniform_score": (
                    track.uniform_score if track.uniform != CHECKING else None
                ),
                "glasses": track.glasses,
                "glasses_score": track.glasses_score,
                "detail": " | ".join(details),
            }
        )
    return rows


def frame_timestamp() -> float:
    return time.monotonic()