# Week 2 Submission

**Student number:** 20000000
**Name:** A. Student

> This sheet is **already filled in** with the answers from the reference run
> (`data/lending_data.csv`, seed 1). Your job is to run each notebook cell and
> **check every number against your own screen**. If a number differs, change it
> to yours. Add your student number and name, then run `python make_submission.py`.
>
> Each answer says **where to look**: the exact printout, table row or column in the
> notebook that shows it. You never need the slides, the booklet, the model card or the
> datasheet to verify Steps 1 to 4.

---

## 0. Environment check

Output of `python setup_check.py` on the reference machine:

```
14 ready · 4 to sort out later · 0 blocking
Nothing blocking.  (shap, lime, ipywidgets and a compiler are the four "later" items; none is used.)
```

---

## Step 1 · Meet the data

**1a.** Share of the dataset that is group B: **0.204** (1,628 of 8,000 people, about one fifth)
*Where to look: Step 1, Table C, row B, column `share`.*

**1b.** Recorded repayment rate: group A **0.547**, group B **0.437**
*Where to look: Step 1, Table C, column `repaid_rate`.*

**1c.** The groups are equally able to repay (true ability −0.017 vs −0.005), so the difference in 1b is caused by **how the data was made**, not ability.
*Where to look: Step 1, Printout B (the two abilities are equal) and Printout A (what made the difference).*

**1d.** "Some group B people who repaid were recorded as defaulting" belongs to pipeline stage **Labels** (the recorded outcome is a record of what the institution wrote down, not what happened).
*Where to look: Step 1, Printout A, the `label_bias` line: that sentence is printed there with its stage.*

---

## Step 2 · Does hiding `group` fix it?

**2a.** Accuracy of guessing `group` from the permitted columns: **99.6 %** (a lazy "always A" guess scores 79.6 %)
*Where to look: Step 2, Line A.*

**2b.** The column that leaks the group: **`postcode_score`** (importance 0.326; every other column is below 0.001)
*Where to look: Step 2, Table C, top row.*

**2c.** Deleting the `group` column does not stop the model treating the groups differently because `postcode_score` carries almost the same information: the model can still tell the groups apart through the proxy, so the group was never really hidden.
*Where to look: 2a and 2b are the evidence.*

---

## Step 3 · The model, and the line you draw

**3a.** At threshold 0.5, approval rate A **0.596**, B **0.393**, gap **0.203**
*Where to look: Step 3, `try_threshold(lab, 0.5)`: rows A and B, column `approval rate`; the gap is on the first printed line.*

**3b.** Another threshold tried: threshold **0.40** → gap **0.19** → overall approval rate **about 70 %**. (At 0.30 the gap is about 0.15 with about 85 % approved; lowering the line approves more people but the gap barely closes.)
*Where to look: the first printed line of your own `try_threshold` run.*

**3c.** At the threshold with the smallest gap (0.98), the share of people who get a loan: **1.9 %** (gap 0.024, accuracy 0.489)
*Where to look: Step 3, `find_equal_approval_threshold` table, top row, column `approval rate overall`.*

**3d.** A fairness number must be reported together with the approval rate because the gap can be made almost zero by approving nearly nobody (0.98) or nearly everybody (0.02); without the approval rate you cannot tell a fair lender from a useless one.
*Where to look: 3c is the evidence.*

---

## Step 4 · Four fixes, and what each costs

**4a.** The fix that made the approval gap *worse*: **drop `postcode_score`** (gap 0.203 → 0.273, because the model leaned harder on the other columns that also leak the group)
*Where to look: Step 4, fixes table, column `selection_gap`: the only row bigger than the baseline's 0.203.*

**4b.** The fix that got the gap to 0.000: **per-group thresholds tuned for the selection gap** (A 0.48, B 0.30), and the other gap that went up: **`ppv_gap`, 0.079 → 0.163** (an approved B applicant is now less likely to actually repay than an approved A applicant)
*Where to look: Step 4, fixes table, the row with `selection_gap` 0.000, then read across to `ppv_gap`.*

**4c.** The fix that worked best overall: **fix the label (stage 3)**, accuracy went from 0.699 to **0.698** (no real cost) while the approval gap fell to **0.033** and every other gap fell too (tpr 0.033, fpr 0.032, ppv 0.009)
*Where to look: Step 4, fixes table, last row, compared with the baseline row.*

**4d.** Fixing the *label* worked better than fixing the *model* because the harm was introduced at the Labels stage of the pipeline; correcting the labels removes it at source, whereas model-side fixes can only move errors from one group to another.
*Where to look: Step 1, Printout A: `label_bias` is a Labels-stage mechanism.*

---

## Step 5 · The mini model card

`model_card.md` exists and all six boxes are filled: **yes** (see `model_card.md` in this folder; `datasheet.md` for the dataset is also completed)
*Where to look: `model_card.md` box 5 and `datasheet.md` "Composition". These two files are only needed for this step.*

---

## Optional

Tried: *Going further* A (mechanism table): `historical_bias` alone gives a gap of 0.148, `label_bias` alone costs the most accuracy (0.662), `proxy` alone does nothing (0.004) until another mechanism gives it something to leak. B (impossibility table): each row zeroes one gap and no row zeroes all three. D: logistic regression is slightly *more* accurate on the worst group (B 0.684 vs 0.668) than the gradient-boosted model.

**Roughly how long did the lab take?** about 2 hours

**Anything that did not work?** Nothing on the reference run. (The earlier edition's plots did not show in Jupyter; fixed.)
