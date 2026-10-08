# Mini Model Card: loan approval model (Week 2)

## 1. What is this model for?
This model gives a score from 0 to 1 for a loan applicant, meaning the estimated
probability that the applicant will repay a personal loan, based on seven permitted
financial features. It is intended as **one input** to a lending decision reviewed by a
human, on applicants similar to the 8,000 in the training file.
It should **not** be used for pricing (setting interest rates), for employment or tenancy
screening, for any decision where the applicant has no route to challenge the outcome, or
for any population where the recorded repayment history was collected under different
rules from the training data.

## 2. What does it actually predict, and what do we really care about?
The column it predicts is `repaid`, which is a record of **what the institution wrote
down**: whether the account was marked as repaid or in default.
What we really care about is whether the person **would actually repay** a loan.
The difference between those two falls hardest on group **B**, because arrears were
recorded more aggressively for group B: some B borrowers who did repay are recorded as
defaulting (label bias, pipeline stage 3). The model learns that record faithfully, so it
under-scores B applicants who are in fact good borrowers.

## 3. What data was it trained on, and who is under-represented?
8,000 synthetic applicants (`data/lending_data.csv`), split 65 % train / 35 % test.
Group B is **20.4 %** of the data (sampling bias), so it exerts little pressure on the
training loss.
The column `postcode_score` is a proxy for group because it is almost perfectly aligned
with group membership (mean 0.717 for A, 0.064 for B): group can be recovered from the
permitted columns at 99.6 % accuracy, with `postcode_score` carrying nearly all of it.
Both groups have identical underlying ability to repay (true ability −0.017 vs −0.005).

## 4. Which fairness criterion did we choose, and why?
We chose to measure fairness as **equal chance that a good borrower is approved between
groups** (equal true-positive rate, "equal opportunity"), with the approval-rate gap
reported alongside it.
We chose this one for a *loan* decision because the harm we most want to avoid is a
creditworthy person being refused credit they would have repaid; that harm is borne by
the applicant, while a wrongly approved loan is borne by the bank, which is better placed
to absorb it. Equal approval rate alone can be satisfied trivially (approve 2 % of
everyone), so it is reported but not optimised.

## 5. What did we do about the gap, and what did it cost?
We applied the fix **"fix the label (stage 3)"** from Step 4: retrain on the corrected
repayment record instead of the biased one.
Result from the reference run: accuracy **0.698** (baseline 0.699); approval-rate gap
**0.033** (baseline 0.203); good-borrower-approval gap **0.033** (baseline 0.205);
false-approval gap **0.032** (baseline 0.122).
The gap that did *not* fully close is the approval-rate gap itself (0.033 remains, from
historical disadvantage in the recorded incomes and savings). The alternatives were
worse: dropping `postcode_score` raised the gap to 0.273; per-group thresholds zeroed one
gap but raised the predictive-value gap from 0.079 to 0.163.

## 6. Who bears each kind of error?
A good borrower who is wrongly refused bears the loss of credit they would have repaid:
a person, disproportionately in group B under the baseline model.
A bad borrower who is wrongly approved means the **bank** bears a financial loss on that
loan.
A person who is refused can challenge the decision by requesting the score, the three
features that most lowered it (from the local explanation), and a review by a human
underwriter who may override the model; every refusal is logged with the model version
and threshold used.
