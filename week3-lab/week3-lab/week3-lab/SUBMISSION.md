# Week 3 Lab: Submission

**Name:** A. Student   **Student number:** 20000000

> This sheet is already filled in with the answers from the reference run. Run each
> notebook cell, find the number where it says to look, and change anything that differs
> on your machine. Replace the name and student number with yours, then upload this file.
> Labs are not graded.

---

## Exercise 1 · A formula with dials  *(lecture Part 1, slides 6 to 11)*

**1a.** With `a=2.0, b=50` the loss is **975.0**.
*Where to look: Printout A, first line.*

**1b.** Table B: loss on trip 0 is **71,255**; on trip 5 it is **106.2**.
*Where to look: Table B, column `loss`, first and last row.*

**1c.** The box that changes `a` and `b` is **Nudge** (the update). Guess and Check only produce the loss; Blame only says which way to turn.
*Where to look: Table B: the dials change between rows because of the nudge.*

**1d.** The step size `0.00003` is a **hyperparameter**: you chose it before training. `a` and `b` are the parameters the computer learned.

**1e.** After 20,000 trips, each extra square metre adds about **2,800 euro** (a ≈ 2.80).
*Where to look: Printout C, second line.*

---

## Exercise 2 · Four shapes of model  *(lecture Part 2, slides 12 to 19)*

**2a.** Highest test score: **SVM (RBF kernel)**, 0.895.
*Where to look: Table A, column `test`.*

**2b.** The two straight-line models: **Logistic regression** and **SVM (linear)**.
*Where to look: Table A, column `shape it draws`.*

**2c.** A decision tree draws **right-angled steps (a staircase)**, because each question asks about one feature at a time ("is x above 3?"), so every split is a vertical or horizontal cut.
*Where to look: the boundaries picture; slide 15.*

**2d.** Each model can only draw its own shape, so picking a model is picking, in advance, what shape you believe the answer has. If the true boundary is a curve, a straight-line model cannot find it however much data you give it.

---

## Exercise 3 · Memorising is not learning  *(lecture Part 3, slides 20 to 23)*

**3a.** Test score is highest at depth **5** (0.890).
*Where to look: Table A, column `test`.*

**3b.** Gap for the unlimited tree: **0.143** (train 1.000, test 0.857).
*Where to look: Table A, last row, or Printout B.*

**3c.** A training score of 1.000 means the tree got every dot it studied right, including the noise; on new dots it is beaten by the depth-3 tree. Perfect on training is the symptom of memorising, not a sign of learning.
*Where to look: Printout B.*

**3d.** The straight line (degree 1) is **underfitting**; the degree-18 curve through every dot is **overfitting** (true error about 1,500). Degree 3 is about right.
*Where to look: the fit picture and its table.*

---

## Exercise 4 · Walking downhill  *(lecture Part 4, slides 24 to 30)*

**4a.** The first three step lengths: **1.05, 0.79, 0.59**. Each is 0.25 × the slope, and the slope shrinks as the walker nears the bottom.
*Where to look: Table A, column `step length`.*

**4b.**

| learning rate | what happened |
|---|---|
| 0.02 | too small (final loss 1.59, still descending) |
| 0.10 | worked |
| 0.90 | worked |
| 1.98 | bounced (oscillating, final loss 1.59) |
| 2.05 | exploded (diverging, loss 396 and rising) |

*Where to look: Table B, column `verdict`.*

**4c.** The flip is at about **2.0**: 1.99 converges, 2.01 diverges. On this bowl the update is x ← (1 − lr)·x, which only shrinks when lr is below 2.
*Where to look: your own `try_lr` runs.*

**4d.** **No error.** The program kept running and simply printed a huge loss. That is dangerous because unless you look at the loss yourself, nothing tells you that training has failed.
*Where to look: Table B, the 2.05 row.*

**4e.** The walker starting at **5.5** ended in the wrong dip (loss 8.95 instead of 6.66). That is a **local minimum**.
*Where to look: Table C, last row.*

---

## Exercise 5 · One neuron, its wall, and the hidden layer  *(lecture Part 5, slides 31 to 37)*

**5a.** The perceptron gets **2 of 4** (some seeds reach 3; never 4).
*Where to look: Table A, first printed line.*

**5b.** The network with one hidden layer gets **4 of 4** (loss about 0.0025).
*Where to look: Table B, first printed line.*

**5c.** Both "1" dots were moved to about **(0.1, 0.03)**: on top of each other, away from the two "0" dots, so one straight line separates them.
*Where to look: Table B, column `hidden layer moved it to`, rows (0, 1) and (1, 0).*

**5d.** With the bend off it outputs about **0.5 for every dot** (2 of 4). Two straight-line layers multiply out to one straight line, and no straight line separates XOR, so the best it can do is sit in the middle.
*Where to look: Table C, column `network says`.*

---

## Exercise 6 · Blame, and the five lines  *(lecture Part 6, slides 38 to 43)*

**6a.** Output **0.626**, target 1.0, loss **0.070**.
*Where to look: Printout A.*

**6b.** `w2` (the output dial) got more blame: **−0.0565** against **−0.0160** for `w1`. Blame shrinks as it is passed backwards through the bend.
*Where to look: Printout B.*

**6c.** By nudging each dial by a hair (0.000001), running forward again, and measuring how much the loss changed. The two numbers matched the blame exactly. That is the gradient check.
*Where to look: Printout C.*

**6d.** `optimiser.zero_grad()` = housekeeping (wipe last time's blame) · `pred = model(X)` = **Guess** · `loss = loss_fn(pred, Y)` = **Check** · `loss.backward()` = **Blame** · `optimiser.step()` = **Nudge**.
*Where to look: the comments printed by `five_lines`.*

**6e.** `loss.backward()` computes the blame (the gradient of the loss with respect to every dial). It does not change any dial; `optimiser.step()` does the nudging.

---

## Optional

Tried: `hidden=8`. The loss falls faster, but the answer is still 4 of 4; with four dots there is nothing more to get right.

**Roughly how long did the lab take?** about 2 hours
