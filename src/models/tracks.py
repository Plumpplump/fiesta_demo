"""Cross-frame person tracking.

The expensive classifiers run once per person, not once per frame: a track is
created when a detection does not overlap an existing track (i.e. somebody new
walked into view), and only unfilled or stale tracks get classified again.
"""

from __future__ import annotations

from . import constants as C
from .geometry import iou, sub_box
from .types import Box, Track


def torso_of(box: Box) -> Box:
    """Middle band of a person box: the torso, without head or legs."""
    return sub_box(
        box,
        top=C.TORSO_TOP,
        bottom=C.TORSO_BOTTOM,
        left=C.TORSO_LEFT,
        right=C.TORSO_RIGHT,
    )


class TrackManager:
    def __init__(
        self,
        iou_threshold: float = C.TRACK_IOU_THRESHOLD,
        max_misses: int = C.TRACK_MAX_MISSES,
    ):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: list[Track] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 1

    def _free_label_index(self) -> int:
        used = {t.label_index for t in self._tracks}
        index = 1
        while index in used:
            index += 1
        return index

    def update(self, boxes: list[Box], now: float) -> tuple[list[Track], list[Track]]:
        """Match boxes to tracks.

        Returns ``(active_tracks, new_tracks)``. ``new_tracks`` are the people
        that appeared in this frame and still need to be classified.
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
            track.torso_box = torso_of(box)
            track.hits += 1
            track.misses = 0
            track.last_seen = now

        self._tracks = [t for t in tracks if t.misses <= self.max_misses]

        new_tracks: list[Track] = []
        for box_index in sorted(free_boxes):
            box = boxes[box_index]
            track = Track(
                track_id=self._next_track_id,
                label_index=self._free_label_index(),
                box=box,
                torso_box=torso_of(box),
                first_seen=now,
                last_seen=now,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            new_tracks.append(track)

        active = [t for t in self._tracks if t.misses == 0]
        return active, new_tracks