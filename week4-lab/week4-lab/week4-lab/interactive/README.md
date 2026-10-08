# Interactive companions, Week 4

Three single-file HTML pages. No build step, no CDN, no network access. Open any
of them directly in a browser, or embed in Tutors:

```html
<iframe src="convolution-feature-map-visualiser.html" width="100%" height="1000" style="border:0"></iframe>
<iframe src="attention-visualiser.html"               width="100%" height="1040" style="border:0"></iframe>
<iframe src="sequence-memory-explorer.html"           width="100%" height="960"  style="border:0"></iframe>
```

Every number these pages display is computed live from real arithmetic. Nothing
is hard-coded or faked. Where a page uses a different random generator from the
Python lab, the digits differ slightly and the note in the source says so.

---

## `convolution-feature-map-visualiser.html`

Booklet chapters 6 to 9. Four tabs.

**Tab 1, watch the kernel walk.** Press **Step** rather than **Walk** for the
first twenty positions. The middle panel shows the patch and the kernel side by
side with the multiplication written out and the running total; the amber square
is where the kernel sits and the amber cell is where the answer lands.

**Tab 2, a bank of filters.** One image, six filters, six channels. Turn ReLU off
and on: with it off each map has both polarities of the same feature, and with it
on half of each map goes flat. That is polarity being separated into distinct
features.

**Tab 3, depth and the receptive field.** Drag *Layers* from 1 to 8 and watch the
boxes grow outward, with the arithmetic in the table beside them. Then set *Pool
every* to 1 and to 4 and compare how fast the field grows.

**Tab 4, break the assumption.** The one to run in the lecture. Choose the
hand-drawn 7 and switch between *shift it by 5 pixels* and *shuffle the pixels*.
Under the shift the edge response moves and survives, because convolution is
equivariant to translation. Under the shuffle it collapses, while the histogram
panel shows every pixel value is still present and unchanged.

**Live demonstration worth doing:** tab 1, set the kernel to *vertical edge* and
press Finish. Note which parts of the image responded. Switch to *horizontal edge*
and finish again. Same image, same nine-number machinery, a completely different
map. The size arithmetic readout confirms `out = (n + 2p - k) / s + 1` as you
change stride and padding.

It reproduces the Python lab's size arithmetic exactly (32, 30, 15 for the three
standard settings) and the receptive-field sequence exactly
(3, 5, 6, 10, 14, 16, 24, 32). The synthetic image's noise uses a different
generator from numpy's, so filter responses differ in the third decimal.

---

## `attention-visualiser.html`

Booklet chapters 17 to 21. Five tabs. Token embeddings are hand-built from
conjunctive grammatical features (plural noun, singular verb, determiner, and so
on) so that the patterns are interpretable without training, but the arithmetic
is exactly `softmax(Q K^T / sqrt(d)) V` with real projection matrices and real
dot products.

**Tab 1, who attends to whom.** Choose the first sentence and the agreement head,
then click **are**. It gives about 0.65 of its attention to **keys**, seven words
back, and far less to **table**, which is adjacent. Drag *Sharpness* down towards
0.2 and watch every row flatten to a uniform average, which is what an untrained
model does.

**Tab 2, one query step by step.** The four stages as bar charts: raw dot
products, scaled, after softmax, and the output vector. The panel on the right
writes the arithmetic out including the check that the weights sum to exactly 1.
Drag sharpness to the far right to see softmax saturation, which is what the
`sqrt(d)` division exists to prevent.

**Tab 3, attention cannot see order.** The most important tab. Press **Shuffle
the tokens**: the third panel is the shuffled result mapped back to the original
order, and it is identical to the first. The readout gives the difference as
`0.00e+0`. Now tick *add positional encoding* and shuffle again.

**Tab 4, many heads.** Four heads on one sentence. Compare head 1 (agreement)
with head 4 (previous word): one is entirely about content and ignores distance,
the other entirely about distance and ignores content. Both are the same
operation with different projections.

**Tab 5, the bill.** Cost and path length against sequence length, with the
crossover computed for whatever model width you choose.

**Live demonstration worth doing:** tab 3, in this order. Shuffle without
positional encoding and read the zero. Ask the room what that means. Then tick
the box. It is the whole justification for positional encoding in one toggle.

---

## `sequence-memory-explorer.html`

Booklet chapters 12 to 16. Three tabs.

**Tab 1, the gradient through time.** The curve is computed live by the same
method the Python lab uses: unroll a real random recurrent matrix, accumulate the
Jacobian, and take its largest singular value by power iteration. Set the scale to
0.90 and the length to 20, and the verdict reads "usable". Now drag the length to
150 **without changing anything else**: nothing about the model changed, and the
task became unlearnable. Then push the scale to 1.1 for the opposite failure.

**Tab 2, what a gate does.** Set the forget gate to 0.5 and the spike at step 3
disappears within a few steps. Drag to 0.95 and it persists to the end. The
half-life table highlights the row you are on. Set the output gate to 0 and note
that the cell fills while the visible state stays flat, which is holding
something in mind without acting on it.

**Tab 3, unrolling and weight sharing.** Drag the sequence length and watch the
parameter table: the recurrent rows do not move, the dense row grows linearly and
can only accept one length. Every `W` in the diagram is the *same* matrix, which
is the most commonly misread thing about recurrent networks.

**Live demonstration worth doing:** tab 1, scale 0.9, then drag the length slider
slowly from 20 to 150. The point lands when students notice that *nothing about
the model changed*.

The browser's random generator differs from numpy's, so the digits differ from
the Python table while the behaviour is identical: at scale 0.5 the norm falls
past 1e-16 within sixty steps, and near scale 1.0 it wanders without a trend,
which is the knife edge.
