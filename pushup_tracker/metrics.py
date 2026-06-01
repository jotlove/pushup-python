"""Front-view push-up depth from multiple pose signals."""

from dataclasses import dataclass
from typing import Optional

from mediapipe.tasks.python.vision import PoseLandmark

from pushup_tracker.geometry import angle_degrees
from pushup_tracker.smooth import Ema


@dataclass(frozen=True)
class RawSignals:
    shoulder_lift: float
    elbow_lift: float
    chest_lift: float
    elbow_angle: float


def _vis_ok(*landmarks, min_visibility: float) -> bool:
    return all(lm.visibility >= min_visibility for lm in landmarks)


def extract_signals(landmarks, min_visibility: float = 0.55) -> Optional[RawSignals]:
    """Raw measurements before smoothing (all in normalized image coords)."""
    pairs_lift = (
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_WRIST),
        (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_WRIST),
    )
    pairs_elbow = (
        (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
        (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    )
    pairs_chest = (
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
        (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    )
    pairs_angle = (
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
        (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    )

    shoulder_lifts: list[float] = []
    elbow_lifts: list[float] = []
    chest_lifts: list[float] = []
    angles: list[float] = []

    for s_i, w_i in pairs_lift:
        s, w = landmarks[s_i.value], landmarks[w_i.value]
        if _vis_ok(s, w, min_visibility=min_visibility):
            shoulder_lifts.append(w.y - s.y)

    for e_i, w_i in pairs_elbow:
        e, w = landmarks[e_i.value], landmarks[w_i.value]
        if _vis_ok(e, w, min_visibility=min_visibility):
            elbow_lifts.append(w.y - e.y)

    for s_i, e_i, w_i in pairs_chest:
        s, e, w = landmarks[s_i.value], landmarks[e_i.value], landmarks[w_i.value]
        if _vis_ok(s, e, w, min_visibility=min_visibility):
            chest_y = (s.y + e.y) / 2.0
            chest_lifts.append(w.y - chest_y)

    for s_i, e_i, w_i in pairs_angle:
        s, e, w = landmarks[s_i.value], landmarks[e_i.value], landmarks[w_i.value]
        if _vis_ok(s, e, w, min_visibility=min_visibility):
            angles.append(
                angle_degrees((s.x, s.y), (e.x, e.y), (w.x, w.y))
            )

    if len(shoulder_lifts) < 1 or len(elbow_lifts) < 1:
        return None

    chest = sum(chest_lifts) / len(chest_lifts) if chest_lifts else sum(shoulder_lifts) / len(shoulder_lifts)
    angle = sum(angles) / len(angles) if angles else 150.0

    return RawSignals(
        shoulder_lift=sum(shoulder_lifts) / len(shoulder_lifts),
        elbow_lift=sum(elbow_lifts) / len(elbow_lifts),
        chest_lift=chest,
        elbow_angle=angle,
    )


class DepthEstimator:
    """
    Fuses shoulder / elbow / chest lift + 2D elbow angle into a 0–1 depth score.

    1.0 ≈ top of push-up (arms extended), 0.0 ≈ bottom.
    """

    WEIGHTS = (0.35, 0.35, 0.15, 0.15)

    def __init__(self, smooth_alpha: float = 0.4) -> None:
        self._smoothers = [Ema(smooth_alpha) for _ in range(4)]
        self._range_min: float | None = None
        self._range_max: float | None = None

    def reset_range(self) -> None:
        self._range_min = None
        self._range_max = None
        for s in self._smoothers:
            s.reset()

    def observe_calibration(self, raw: RawSignals) -> float:
        """Record top-position samples; returns smoothed combined depth (un-normalized)."""
        return self._combined(raw)

    def _combined(self, raw: RawSignals) -> float:
        parts = (
            raw.shoulder_lift,
            raw.elbow_lift,
            raw.chest_lift,
            raw.elbow_angle / 180.0,
        )
        smoothed = [s.update(p) for s, p in zip(self._smoothers, parts)]
        w = self.WEIGHTS
        return w[0] * smoothed[0] + w[1] * smoothed[1] + w[2] * smoothed[2] + w[3] * smoothed[3]

    def update(self, raw: RawSignals) -> tuple[float, float | None]:
        """
        Returns (combined_depth, depth_percent).

        depth_percent is 0–1 when range is known, else None.
        Range widens only when you clearly hit a new top or bottom.
        """
        combined = self._combined(raw)
        if self._range_min is None or self._range_max is None:
            return combined, None

        span = self._range_max - self._range_min
        if span < 0.045:
            return combined, None

        depth_pct = (combined - self._range_min) / span

        if depth_pct < 0.12:
            self._range_min = 0.9 * self._range_min + 0.1 * combined
        elif depth_pct > 0.88:
            self._range_max = 0.9 * self._range_max + 0.1 * combined

        span = self._range_max - self._range_min
        if span < 0.045:
            return combined, None

        depth_pct = max(0.0, min(1.0, (combined - self._range_min) / span))
        return combined, depth_pct

    def set_range_from_calibration(self, samples: list[float]) -> None:
        if not samples:
            return
        ordered = sorted(samples)
        # User holds top position: treat high percentiles as "up"
        self._range_max = ordered[int(len(ordered) * 0.85)]
        self._range_min = ordered[int(len(ordered) * 0.15)]
        span = self._range_max - self._range_min
        if span < 0.03:
            self._range_min = self._range_max - 0.08

    @property
    def range_span(self) -> float:
        if self._range_min is None or self._range_max is None:
            return 0.0
        return self._range_max - self._range_min
