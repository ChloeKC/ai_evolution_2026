"""
optim.py, one optimiser, shared by all three model families in this lab.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

Week 3 built gradient descent and then momentum, and ended with a warning:
every fix creates the next problem. Adam is the fix that most people use, and
this file exists so that the three architectures in this lab are compared
FAIRLY. If the attention model were given Adam and the recurrent models were
left on plain descent, the comparison would be measuring the optimiser and
reporting it as a fact about architectures.

Adam, in one sentence: keep a running average of each parameter's gradient
(that is momentum, from Week 3), keep a running average of the SQUARE of each
parameter's gradient, and divide the first by the square root of the second.

Why the division helps. Gradient descent uses one learning rate for every
parameter, so the step size has to suit the parameter with the largest
gradient, and every parameter with a small gradient then crawls. Dividing by
each parameter's own typical gradient size removes that coupling: a parameter
whose gradients are consistently tiny still takes a full-sized step, because
"full-sized" is now measured relative to itself.

This matters concretely for Week 4. To attend sharply, a model must push its
query and key projections to large values so the dot products spread out and
the softmax concentrates on one position. Under plain descent at a sane
learning rate that takes a very long time. Before Adam was added to this lab,
the attention model scored 0.448 on the length-13 lookup task; with Adam and
nothing else changed it scores 0.998. The architecture was never the problem.

The gradient clipping is a separate patch for a separate problem, and it comes
from Chapter 14: a recurrent network's gradient can explode, and clipping
rescales any gradient whose length exceeds a threshold. It caps the size and
keeps the direction.
"""

from __future__ import annotations

import numpy as np


class AdamMixin:
    """Adds `step()` to any class that lists its parameter names in `self.params`
    and stores gradients as `self.d<name>`."""

    def step(self, lr, clip=5.0, b1=0.9, b2=0.999, eps=1e-8):
        if not hasattr(self, "_m"):
            self._m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
            self._v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
            self._t = 0
        self._t += 1
        for p in self.params:
            g = getattr(self, "d" + p)
            nrm = np.linalg.norm(g)
            if clip is not None and nrm > clip:
                g = g * (clip / nrm)          # cap the size, keep the direction
            self._m[p] = b1 * self._m[p] + (1 - b1) * g
            self._v[p] = b2 * self._v[p] + (1 - b2) * g * g
            mhat = self._m[p] / (1 - b1 ** self._t)      # bias correction, because
            vhat = self._v[p] / (1 - b2 ** self._t)      # both averages start at 0
            setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))
