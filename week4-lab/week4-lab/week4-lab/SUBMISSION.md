# Week 4 Lab: Submission

**Name:** A. Student   **Student number:** 20000000

> This sheet is already filled in with the answers from the reference run. Run each
> notebook cell, find the number where it says to look, and change anything that differs
> on your machine (small differences in the third decimal are normal). Replace the name and
> student number with yours, then upload this file. Labs are not graded.

---

## Exercise 1 · A guess about the world  *(lecture Part 1, slides 7 to 13)*

**1a.** All **five** rules fit the three examples. The examples cannot choose between them.
*Where to look: Printout A.*

**1b.** At 128 pictures: tiny model (700 dials) test **0.941**; big model (16,708 dials) test **0.264**, which is chance.
*Where to look: Table B, column `test`.*

**1c.** **No.** The big model scored **1.000** on the pictures it studied. It was not short of dials or training; it was short of reasons to prefer the right answer.
*Where to look: Table B, column `train`.*

**1d.** After the shuffle the tiny model scores **0.255** (chance), and the gap flips from **+0.664** to **−0.078**.
*Where to look: Table C, row "pixels shuffled".*

**1e.** The tiny model's whole advantage was one guess, that neighbouring pixels belong together; destroying only that guess destroyed the whole advantage.
*Where to look: Table C.*

---

## Exercise 2 · Convolution  *(lecture Part 2, slides 14 to 20)*

**2a.** Dark-to-bright patch: **3.2**. Flat patch: **0.0**. Reversed patch: **−3.2**.
*Where to look: Printouts B, C, D.*

**2b.** The kernel's nine numbers add up to zero, so a patch that is the same everywhere multiplies out to zero. The kernel detects *change*, not brightness.
*Where to look: Printout A.*

**2c.** The 4s sit in the two middle columns of the feature map, exactly where the picture changes from dark to bright. Big numbers mark *where* the pattern is.
*Where to look: Table B.*

**2d.** Convolution: **0.00e+00**. Dense layer: **1.987**.
*Where to look: Table C, column `value`.*

**2e.** The same nine numbers are used at every position, so the operation has no way of knowing where it is; shifting the picture just shifts which position produces which number.
*Where to look: red box 2.*

---

## Exercise 3 · What the guess buys  *(lecture Part 3, slides 21 to 25)*

**3a.** Layer 2: **25** pixels (5 by 5). Layer 8: **1,024** pixels (32 by 32).
*Where to look: Table A, column `pixels seen`.*

**3b.** Averaging ending: **0.998**. Flatten ending: **0.230**.
*Where to look: Table B, column `positions never seen`.*

**3c.** The flatten ending wired a separate dial to every position, so a pattern on the right went through dials that had never been trained; the averaging ending does not care where the pattern is.
*Where to look: Table B.*

**3d.** The plain model won (**0.978** vs **0.940**). The digits were already centred, so "where is it?" was solved before the model saw the data and the convolution's guess had nothing left to buy. That is the week's point, not an exception to it.
*Where to look: Table C, column `verdict`.*

---

## Exercise 4 · Recurrence  *(lecture Part 4, slides 26 to 35)*

**4a.** 0.9 to the sixtieth: **0.0018**. 1.1 to the sixtieth: **304**.
*Where to look: Printout A.*

**4b.** At scale 0.5: **6.96e-17** (zero to a computer). At 0.9: **1.49e-01**.
*Where to look: Table B.*

**4c.** With keep-gate 0.95, half the memory is left after **13.5** steps.
*Where to look: Table C.*

**4d.** At 90 steps: plain chain **0.463** with signal **7.8e-18**; gated **1.000** with signal **8.9e-06**.
*Where to look: Table D, row 90.*

**4e.** In the plain chain the signal is multiplied by the *same* number at every step, so it dies or explodes; in the gated one it is multiplied only by a keep-gate the network chooses, and it learns to keep the gate near 1.
*Where to look: red box 4.*

---

## Exercise 5 · Attention  *(lecture Part 5, slides 36 to 45)*

**5a.** Row "are": **0.735** on "keys", **0.013** on "left".
*Where to look: Table A, row `are`.*

**5b.** Each row sums to **1** (1.00 to rounding).
*Where to look: Table A, column `sum`.*

**5c.** Write the cards (query, key, value); score every query against every key; share out each row with softmax; mix the values by those shares.
*Where to look: Printout B.*

**5d.** Without position labels: **8.88e-16** (rounding). With: **4.192**.
*Where to look: Table C.*

**5e.** At 32,768 words attention costs **65 times** the chain; its longest path is **1** (the chain's is 32,768).
*Where to look: Table D, last row.*

---

## Exercise 6 · The bargain  *(lecture Part 6, slides 46 to 50)*

**6a.** At 64 pictures: **+47.2 points**. At 512: **+66.3 points**.
*Where to look: Table A, column `CNN advantage`.*

**6b.** **1.000** at every size.
*Where to look: Table A, column `MLP train`.*

**6c.** The bargain flips when there is enough data for the machine that guesses less to learn the structure from examples (the slide: about 128 times the data; the Vision Transformer: 300 million pictures). What flips it is the amount of data, nothing else.
*Where to look: red box 6.*

**6d.** Question **4**: is the sentence true of my data, and how would I know?
*Where to look: Printout B.*

**6e.** Question **5**: does my ending keep the guess all the way to the answer?
*Where to look: Printout B.*

---

## Optional

Tried: the attention visualiser, tab 3. Shuffling the tokens gives exactly the same output without position labels, as Table C said.

**Roughly how long did the lab take?** about 2 hours
