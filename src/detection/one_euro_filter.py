"""
One Euro Filter Implementation.
A precise, adaptive low-pass filter with dynamic cutoff frequency.
Minimizes jitter at low movement speeds while eliminating phase lag during fast motions.
Reference: Casiez, G., Roussel, N. and Vogel, D. (CHI 2012).
"""

import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    """Standard exponential smoothing filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y: Optional[float] = None
        self.s: Optional[float] = None

    def reset(self) -> None:
        self.y = None
        self.s = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s


class OneEuroFilter:
    """
    1€ Filter for 1D scalar signal.
    Dynamically tunes cutoff frequency based on rate of change (speed).
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 15.0,
        d_cutoff: float = 1.0
    ):
        """
        Args:
            min_cutoff (float): Minimum cutoff frequency in Hz (jitter suppression at low speeds).
            beta (float): Speed coefficient for normalized coordinates (lag reduction at high speeds).
            d_cutoff (float): Cutoff frequency for derivative filtering in Hz.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def reset(self) -> None:
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None

    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filters input sample x.
        
        Args:
            x (float): Noisy raw sample.
            timestamp (float): Current timestamp in seconds.
            
        Returns:
            float: Filtered sample with zero-lag adaptation.
        """
        now = timestamp if timestamp is not None else time.time()

        if self.last_time is None:
            self.last_time = now
            return self.x_filt.filter(x, 1.0)

        dt = now - self.last_time
        self.last_time = now

        # Prevent division by zero or negative dt
        if dt <= 1e-5:
            dt = 1e-4

        # Compute rate of change (derivative)
        prev_x = self.x_filt.s if self.x_filt.s is not None else x
        dx = (x - prev_x) / dt

        # Filter the derivative
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        edx = self.dx_filt.filter(dx, a_d)

        # Dynamic cutoff frequency: increases when speed |edx| increases
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the signal with the adaptive cutoff
        a = self._smoothing_factor(dt, cutoff)
        return self.x_filt.filter(x, a)


class OneEuroFilter2D:
    """Applies independent OneEuroFilters to 2D coordinates (x, y)."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 15.0,
        d_cutoff: float = 1.0
    ):
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def set_parameters(self, min_cutoff: float, beta: float) -> None:
        """Updates filter responsiveness settings."""
        self.fx.min_cutoff = min_cutoff
        self.fx.beta = beta
        self.fy.min_cutoff = min_cutoff
        self.fy.beta = beta

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """Filters 2D point coordinates."""
        now = timestamp if timestamp is not None else time.time()
        return (self.fx.filter(x, now), self.fy.filter(y, now))
