# Week 5 Lab: Submission

**Name:**   **Student number:**

> **This sheet is blank on purpose.** Fill it in as you go through the notebook. Each question is
> one of five kinds:
>
> - **Predict**: write it *before* you run the cell. A wrong prediction is fine; it is the most
>   useful line on the sheet, because it shows you exactly what to rethink. Do not change it
>   afterwards. Say whether you were right in the next question instead.
> - **Record**: the number from the printout the question names.
> - **Explain**: two or three sentences in your own words. No need to quote the slides.
> - **Build**: paste the line of code that passed its check.
> - **Change**: predict, change one thing, run, and say what happened.
>
> Upload this file when you are done, **then** open *Week 5 Reference Answers* on Tutors and
> compare. Labs are not graded: what you write here is for you, and it shows us where the
> material needs work.

---

## Exercise 1 · A law is not a pattern  *(lecture Part 1, slides 7 to 10)*

**1a · Predict.** One *pattern* and one *law* from your own work or life. What does each one do to
the list of possible answers? *(If Printout A changes your mind, add a line saying what you changed.)*

- Pattern:
- Law:

**1b · Predict.** In the race, which ends with the lower error: data-only with 64 measurements, or
physics-informed with none? By roughly what factor: 2, 20 or 200 times? Why?

>

**1c · Record.** Test RMSE, physics-informed: ______   data-only with 64: ______
Was 1b right?

>

**1d · Explain.** What did each model's loss compare the network's guess against? Use that to
explain why one model needs measurements and the other does not.

>

**1e · Change** *(optional)*. Prediction first: with 64 *perfect* measurements, will the data-only
model beat the physics-informed one?   Prediction: ______
Result (Table B2, 64 measurements): ______   If noise was not the whole story, where does the
data-only model's error come from?

>

---

## Exercise 2 · Checking the equation where there is no data  *(lecture Part 2, slides 13 to 17)*

**2a · Predict.** Put each candidate back into m u'' + c u' + k u. Which leaves exactly zero, and
which leaves the most: the true solution, cos(t), or a flat line at 0.7?

>

**2b · Build.** Your `my_residual` line, after every check showed ✓:

```python

```

**2c · Explain.** The flat line leaves the same number at every time. Which number, and why exactly
that one?

>

**2d · Build.** Your `my_physics_loss` line, after every check showed ✓:

```python

```

Why can the loss not simply be the plain average of the residual?

>

**2e · Record and explain.** Your network's test RMSE (Table G): ______   The physics-informed row of
Exercise 1, Table B: ______
What did your two lines have to get right for these to match?

>

**2f · Explain** *(write it before running `one_thing_changed`)*. Which step of Week 3's loop (guess,
check, blame, nudge) did your code replace? Why does that step need no measurements?

>

**2g · Explain** *(optional)*. Which derivative does the finite difference get more wrong (Printout C),
and why does that matter for a loss built on u''?

>

---

## Exercise 3 · What the law buys  *(lecture Part 3, slides 21 to 24)*

**3a · Predict.** Degradation factor beyond the data (about 1, 3, 10 or 30?):
data-only ______   physics plus the same data ______

**3b · Record.** Table A, last column: data-only ______   physics plus data ______   Was 3a right?

>

**3c · Explain.** Why does the physics model keep working where it never saw data?

>

**3d · Change** *(optional)*. Data only up to 20% of the run. Prediction first: whose factor changes
more, and which way?   Prediction: ______
Result: data-only ______   physics plus data ______   What happened, and why?

>

**3e · Predict.** No friction, so energy should stay constant. Rank from most energy drift to least:
data-only; physics plus data; physics only (no data).

>

**3f · Record and explain.** Drift, data-only: ______   physics only: ______
Energy was in nobody's loss. How did the physics-only model keep it constant?

>

---

## Exercise 4 · What it costs, and how it goes wrong  *(lecture Part 4, slides 27 to 34)*

**4a · Predict.** Soft penalty with weight 1, 50 or 500: which gives the lowest error? Will the
weight-500 model get u(0) = 1 right?

>

**4b · Explain.** Table A: how can the weight-500 model get the start right and still be badly wrong?

>

**4c · Build.** Your `my_guess` line, after every check showed ✓ (paste every version you trained):

```python

```

**4d · Record and explain.** Your test RMSE (Table A2): ______   The hard-constraint row of Table A: ______
If yours is close: why does your guess work? If it is much worse: why, using the printout's clue?

>

**4e · Predict.** Checkpoints 10, 50 or 200: which run *reports* the smallest physics loss? Which has
the smallest *real* error?

>

**4f · Explain.** What does a small physics loss actually promise, and what does it leave unchecked?

>

**4g · Change.** The friction you told the model: ______ (the truth is 0.2). Prediction first: will
the physics model beat data-only, match it, or lose?   Prediction: ______
Result, true RMSE: your value ______   data-only ______

**4h · Explain.** At c = 0.8, would the reported physics loss have warned you? Why is a wrong law
worse than no law?

>

---

## Exercise 5 · Four more ways to build in science  *(lecture Part 5, slides 37 to 43)*

**5a · Predict.** Rotate the arrangement: whose answer changes, and by roughly how much?

>

**5b · Record.** Table A, change under rotation: coordinates ______   distances ______
Table B, degradation on unseen orientations: coordinates ______   distances ______

**5c · Predict.** With gravity added, which model wins? Would more data rescue the other?

>

**5d · Explain.** Table C: why can the losing model never catch up?

>

**5e · Predict.** Run three times past the training window: does the model of the *answer* or the
model of the *rule* degrade more?

>

**5f · Record and explain.** Table D, degradation: answer ______   rule ______
Table E: what does the operator trade for its speed?

>

**5g · Explain: teach it back.** One sentence each, for a colleague who missed the lecture:

- Symmetry:
- Neural ODE:
- Neural operator:
- Diffusion (which half is physics, which half is learned):

---

## Exercise 6 · Choosing  *(lecture Part 6, slides 46 to 49)*

**6a · Predict.** Heat along a rod: physics with 0 measurements, or data-only with 128? By roughly
how much?

>

**6b · Record.** Table A: physics-informed ______   data-only with 128 ______

**6c · Match, then mark.** Write your matches *before* running the reveal cell.

| situation | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| your method (A to G) | | | | | | | |

Score after the reveal: ___ / 7. For one you missed, why is the reference right?

>

**6d · Apply.** A problem you know (work, a hobby, or your capstone idea) where some rule is known:

- The problem:
- Q1, what do I know, in plain words:
- Q2, exactly true, roughly true, or a guess:
- Q6, what if my rule is 20% wrong:
- Q7, would a classical solver or ordinary machine learning do better:

---

## Before you upload

**R1.** Which prediction were you most wrong about? What had you assumed, and what do you think now?

>

**R2.** Roughly how long did the lab take?

>
