from enum import Enum, auto


class Phase(Enum):
    UP = auto()
    DOWN = auto()


class CounterState(Enum):
    CALIBRATING = auto()
    TRACKING = auto()


class PushUpCounter:
    """
    Rep counter driven by depth_percent (0 = bottom, 1 = top).

    Uses frame debouncing and a minimum depth to avoid false reps.
    """

    def __init__(
        self,
        down_depth: float = 0.38,
        up_depth: float = 0.72,
        min_range: float = 0.07,
        frames_to_down: int = 4,
        frames_to_up: int = 5,
        calibration_frames: int = 40,
    ) -> None:
        self.down_depth = down_depth
        self.up_depth = up_depth
        self.min_range = min_range
        self.frames_to_down = frames_to_down
        self.frames_to_up = frames_to_up
        self.calibration_frames = calibration_frames

        self.state = CounterState.CALIBRATING
        self.phase = Phase.UP
        self.reps = 0
        self._calibration_samples: list[float] = []
        self._calibration_progress = 0
        self._down_streak = 0
        self._up_streak = 0
        self._deepest_in_rep = 1.0

    def reset(self) -> None:
        self.state = CounterState.CALIBRATING
        self.phase = Phase.UP
        self.reps = 0
        self._calibration_samples.clear()
        self._calibration_progress = 0
        self._down_streak = 0
        self._up_streak = 0
        self._deepest_in_rep = 1.0

    @property
    def is_calibrating(self) -> bool:
        return self.state is CounterState.CALIBRATING

    @property
    def calibration_progress(self) -> float:
        return min(1.0, self._calibration_progress / self.calibration_frames)

    def finish_calibration(self, depth_estimator) -> None:
        depth_estimator.set_range_from_calibration(self._calibration_samples)
        self.state = CounterState.TRACKING
        self.phase = Phase.UP

    def update(
        self,
        depth_pct: float | None,
        range_span: float,
        combined_depth: float,
    ) -> None:
        if self.state is CounterState.CALIBRATING:
            self._calibration_samples.append(combined_depth)
            self._calibration_progress += 1
            return

        if depth_pct is None:
            return

        if self.phase is Phase.UP:
            if depth_pct < self.down_depth:
                self._down_streak += 1
                if self._down_streak >= self.frames_to_down:
                    self.phase = Phase.DOWN
                    self._deepest_in_rep = depth_pct
                    self._up_streak = 0
            else:
                self._down_streak = 0
        else:
            self._deepest_in_rep = min(self._deepest_in_rep, depth_pct)
            if depth_pct > self.up_depth:
                self._up_streak += 1
                went_deep_enough = (1.0 - self._deepest_in_rep) >= self.min_range * 0.85
                range_ok = range_span >= self.min_range
                if (
                    self._up_streak >= self.frames_to_up
                    and went_deep_enough
                    and range_ok
                ):
                    self.phase = Phase.UP
                    self.reps += 1
                    self._down_streak = 0
                    self._up_streak = 0
                    self._deepest_in_rep = 1.0
            else:
                self._up_streak = 0
