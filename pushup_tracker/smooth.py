class Ema:
    """Exponential moving average filter for noisy pose landmarks."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def reset(self) -> None:
        self.value = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value
