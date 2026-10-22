# Interactive companions, Week 5

Three single-file HTML pages. No build step, no CDN, no network access. Open
directly in a browser, or embed in Tutors:

```html
<iframe src="pinn-sandbox.html" width="100%" height="1080" style="border:0"></iframe>
<iframe src="symmetry-explorer.html" width="100%" height="900" style="border:0"></iframe>
<iframe src="diffusion-thermodynamics-explorer.html" width="100%" height="980" style="border:0"></iframe>
```

---

## `pinn-sandbox.html`

Booklet chapters 8 to 12. **This page trains a real physics-informed network in
your browser.** It is not a replay of precomputed curves: the script contains a
reverse-mode automatic differentiation engine of about 120 lines, deliberately
mirroring `autodiff.py`, so the second derivative in the residual is computed the
same way the Python lab computes it.

Verified against the Python lab: the hard constraint gives `u(0) = 1.0000000000`
and `u'(0) = 0.0000000000` exactly, and a full training run reaches an error of
about 1.5e-03 against the lab's 1.35e-03.

**Tab 1, train a PINN.** Press *Train* and watch two numbers: the *physics loss*
being minimised, and the *true RMSE* against a solution the training never sees.
They fall together, which is the whole claim of the week in about fifteen seconds.

*The demonstration to run in the lecture:* drag **collocation points** down to
about 10 and retrain. The physics loss still falls to a small value, because the
equation genuinely is satisfied at those ten places, and the true error is large.
The loss is not lying; it is answering a different question from the one you
meant to ask.

Then untick **impose it exactly** and retrain. Note that u = 0 satisfies the
equation perfectly, so with a soft penalty the only thing standing between the
optimiser and nonsense is the penalty term.

**Tab 2, physics against data.** Both models get the same measurements, on the
left portion of the interval only. Drag the split slider left and press *Train
both*. The two "where data was" numbers stay similar; the two "beyond it" numbers
diverge.

**Tab 3, when the physics is wrong.** The most important tab. Set the assumed
damping away from its true value of 0.20 and train. The error against the truth
climbs while the two numbers you would actually be monitoring, the physics loss
and the data loss, stay small. Nothing goes red.

---

## `symmetry-explorer.html`

Booklet chapters 14 and 15. Everything is computed live from the actual geometry,
and the "coordinate model" is a real network with fixed random weights, so its
non-invariance is genuine rather than staged.

**Tab 1, rotate the molecule.** Press *Spin*. The right-hand panel shows the six
raw coordinates all changing while the three pairwise distances stay identical to
four decimal places. The two readouts show one model's prediction wandering and
the other's not moving at all. Neither is trained: the difference is entirely in
what they read.

Then drag the **translation** slider and watch the same thing happen for a second
symmetry.

**Tab 2, a full turn.** Predicted energy through 360 degrees. The truth is a flat
line, because rotation cannot change energy. Anything that is not flat is an error
the model did not have to make.

**Tab 3, what symmetry costs and buys.** Drag the **gravity** slider up from zero.
At zero the symmetry holds and imposing it is free. Above zero the panel states
plainly that an invariant model cannot represent the target at any dataset size.

---

## `diffusion-thermodynamics-explorer.html`

Booklet chapters 20 and 21.

**One honest note.** In a real diffusion model a network learns the score. On this
page the data is a mixture of Gaussians, and for that family the score of the
noised distribution can be written down *exactly*, so the page uses the exact
score rather than a trained approximation. That is deliberate: it separates two
questions usually tangled together, namely what the method does and how well a
network can approximate it. `diffusion.py` does the other half by training a real
denoiser on data with no closed form.

**Tab 1, forwards into equilibrium.** Press *Run forwards* and watch two crescents
dissolve. The signal-fraction curve tracks how much structure is left. Nothing on
this tab is learned, which is the point: the forward process is given by physics.

**Tab 2, backwards out of noise.** Press *Generate*, then *Animate* to watch it
happen. Drop the reverse steps to 5 and regenerate: the samples blur, because each
step now has to undo more than it can. Set the last slider to 0 and the cloud
collapses onto a few spots, which is why noise is injected on the way out.

**Tab 3, the score.** Drag the step slider from 199 down to 0 and watch the arrow
field sharpen. Early on the density is nearly Gaussian and the arrows point
roughly inward, so coarse structure is recovered first; late on they resolve fine
detail. A diffusion model builds an image the way a photograph develops, and this
tab shows that it is forced by the mathematics rather than chosen.
