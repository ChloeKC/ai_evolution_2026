# Interactive companions, Week 3

Two single-file HTML pages. No build step, no CDN, no network access. Open
either one directly in a browser, or embed it in Tutors:

```html
<iframe src="gradient-descent-explorer.html" width="100%" height="720" style="border:0"></iframe>
<iframe src="hidden-layer-playground.html"   width="100%" height="880" style="border:0"></iframe>
```

## `gradient-descent-explorer.html`

Booklet chapters 11 to 13. Drag the ball to set a starting point, then move the
learning rate slider and watch the verdict change.

Four surfaces: a quadratic bowl, a double well with a genuine local minimum, a
bumpy bowl, and a cliff with a near-vertical face. A second tab walks the same
algorithm across a two-dimensional Rosenbrock contour map, where momentum makes
a visible difference.

It reproduces the Python lab exactly. At `lr = 0.02, 0.1, 0.9, 1.98, 2.05` from
`x = 4`, after 40 steps, both give `1.783 / 0.059 / 4e-40 / 1.783 / 28.16`.

**Live demonstration worth doing:** start on the bowl and raise the learning rate
slowly. Nothing dramatic happens until you pass 2, at which point it diverges.
That threshold is derivable: the update is `x <- (1 - lr) * x`.

## `hidden-layer-playground.html`

Booklet chapters 10, 14 and 15. A 2-h-1 network trained live in the browser by
the same backpropagation you wrote in `scratch_net.py`.

Three panels: the network with weights thickening as it learns, the decision
surface in input space, and the **hidden representation**, which is the panel
that matters.

**Three settings, three lessons:**

| Setting | What happens |
|---|---|
| Hidden units 0, XOR | Sticks at 2 of 4, loss flat at 0.25. The 1969 result, live. |
| Hidden units 2, activation `none` | Still fails. Stacked linear layers collapse to one linear layer. |
| Hidden units 2, activation `sigmoid` | 4 of 4, loss 0.0009. Watch the right-hand panel. |

That last figure matches `scratch_net.py`, which reaches 0.000905.

In the third case, watch the two points that should output 1 slide on top of
each other in the hidden representation panel, while the two that should output
0 move to opposite corners. The green dashed line is the output layer. The
hidden layer did not add power; it changed the coordinates.
