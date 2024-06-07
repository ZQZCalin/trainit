"""Wandb logging functions."""

import time
import wandb
from collections import defaultdict


class EMA:
    """Implements an Exponential Moving Average (EMA) object.
    
    Given a sequence :math:`x_0,\ldots,x_n`, EMA computes the exponential average
    ..math:: \\text{EMA}_n = \\frac{\sum_{t=0}^n \\beta_t x_t}{\sum_{t=0}^n \\beta_t}.

    Let :math:`S_n = (1-\\beta)\sum_{t=0}^n \\beta_t x_t`. 
    In every step, we update :math: `S_t = \\beta S_{t-1} + (1-\\beta) x_t`
    and :math:`\\text{EMA}_t = S_t / (1-\\beta^{t+1})`.
    """
    def __init__(self, window_size):
        self.A = 0.0
        self.beta = 1.0 - 1.0 / window_size
        self.count = 0

    def update(self, value):
        self.A = self.A * self.beta + (1.0 - self.beta) * value
        self.count += 1

    @property
    def value(self):
        return self.A / (1.0 - self.beta ** (self.count + 1))
    

class TimeKeeper:
    """Implements a TimeKeeper object.
    
    Q: what does this do?
    """
    def __init__(self, window_size=100):
        self.timestamps = {}
        # Using defaultdict, any new key will be initialized with an EMA object.
        self.average_durations = defaultdict(lambda: EMA(window_size))
        self.periods = defaultdict(lambda: EMA(window_size))

    def mark(self, start_events=[], end_events={}):
        cur_time = time.time()
        for e, c in end_events.items():
            if c > 0:
                delta = (cur_time - self.timestamps[e]) / c
                self.average_durations[e].update(delta)
        for s in start_events:
            if s in self.timestamps:
                delta = cur_time - self.timestamps[s]
                self.periods[s].update(delta)
            self.timestamps[s] = cur_time

        return cur_time

    def get_durations(self):
        return {k: v.value for k, v in self.average_durations.items()}

    def get_proportions(self):
        return {
            k: self.average_durations[k].value / self.periods[k].value
            for k in self.periods
        }


class RateLimitedWandbLog:
    """Implements a Wandb logger object.
    """
    def __init__(self, max_frequency=1.0):
        self.max_frequency = max_frequency
        self.last_time = time.time() - 1.0 / self.max_frequency
        self.metrics = {}

    def __call__(self, metrics, *args, commit=True, **kwargs):
        self.metrics.update(metrics)
        if commit:
            cur_time = time.time()
            if cur_time >= self.last_time + 1.0 / self.max_frequency:
                wandb.log(self.metrics, *args, **kwargs)
                self.last_time = cur_time
                self.metrics = {}