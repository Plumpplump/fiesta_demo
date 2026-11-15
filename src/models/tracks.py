# Author: David Li, SISHK
"""Cross-frame person tracking.

The expensive classifiers run once per person, not once per frame: a track is
created when a detection does not overlap an existing track (i.e. somebody new
walked into view), and only unfilled or stale tracks get classified again.
"""

from __future__ import annotations

from . import constants as C
from .geometry import iou
from .types import Box, Track


def torso_of(box: Box, frame_height: int | None = None) -> Box:
    """Band of a person box that holds the shirt.

    A close-up webcam box is head + shoulders, so the shirt sits low
    (40-92% of the box); a full-body box has it in the upper-middle (12-62%).
    Using one fixed band crops the face on close-ups, which is the biggest
    accuracy killer measured in the sibling project.
    """
    box_height = box[3] - box[1]
    close_up = (
        frame_height is not None
        and frame_height > 0
        and box_height / float(frame_height) >= C.TORSO_CLOSE_UP_RATIO
    )
    top, bottom = (
        (C.TORSO_TOP_CLOSE, C.TORSO_BOTTOM_CLOSE)
        if close_up
        else (C.TORSO_TOP_FULL, C.TORSO_BOTTOM_FULL)
    )
    # A little margin outside the box keeps the shoulders/sleeves in the crop.
    x1, y1, x2, y2 = box
    margin = C.TORSO_SIDE_MARGIN * (x2 - x1)
    return (x1 - margin, y1 + top * box_height, x2 + margin, y1 + bottom * box_height)


class TrackManager:
    def __init__(
        self,
        iou_threshold: float = C.TRACK_IOU_THRESHOLD,
        max_misses: int = C.TRACK_MAX_MISSES,
        confirm_hits: int = C.TRACK_CONFIRM_HITS,
    ):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.confirm_hits = confirm_hits
        self._tracks: list[Track] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def visible(self) -> list[Track]:
        """Confirmed tracks seen in the current frame (what the preview draws)."""
        return [t for t in self._tracks if t.confirmed and t.misses == 0]

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 1

    def _free_label_index(self) -> int:
        used = {t.label_index for t in self._tracks}
        index = 1
        while index in used:
            index += 1
        return index

    def update(
        self,
        boxes: list[Box],
        now: float,
        frame_height: int | None = None,
    ) -> tuple[list[Track], list[Track]]:
        """Match boxes to tracks.

        Returns ``(active_tracks, new_tracks)``. ``active_tracks`` are the
        confirmed people visible in this frame; ``new_tracks`` still need a
        classification once they collect ``confirm_hits`` detections.
        """
        tracks = self._tracks
        assigned: dict[int, int] = {}
        free_boxes = set(range(len(boxes)))

        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            for box_index, box in enumerate(boxes):
                score = iou(track.box, box)
                if score >= self.iou_threshold:
                    pairs.append((score, track_index, box_index))
        pairs.sort(key=lambda item: item[0], reverse=True)

        for _score, track_index, box_index in pairs:
            if track_index in assigned or box_index not in free_boxes:
                continue
            assigned[track_index] = box_index
            free_boxes.discard(box_index)

        for track_index, track in enumerate(tracks):
            box_index = assigned.get(track_index)
            if box_index is None:
                track.misses += 1
                continue
            box = boxes[box_index]
            track.box = box
            track.torso_box = torso_of(box, frame_height)
            track.hits += 1
            track.misses = 0
            track.last_seen = now
            track.confirmed = track.hits >= self.confirm_hits

        self._tracks = [t for t in tracks if t.misses <= self.max_misses]

        new_tracks: list[Track] = []
        for box_index in sorted(free_boxes):
            box = boxes[box_index]
            track = Track(
                track_id=self._next_track_id,
                label_index=self._free_label_index(),
                box=box,
                torso_box=torso_of(box, frame_height),
                confirmed=self.confirm_hits <= 1,
                first_seen=now,
                last_seen=now,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            new_tracks.append(track)

        active = [t for t in self._tracks if t.confirmed and t.misses == 0]
        return active, new_tracks