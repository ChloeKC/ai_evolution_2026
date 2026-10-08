# Week 2 Lab: Responsible AI

**The Evolution of Machine Learning and AI · SETU**

Five steps. About 20 minutes each. Every step tells you **what to run**, **what you
should see**, and **what to write down**. The steps follow the lecture's pipeline
picture: *World → Data → Labels → Model → Threshold → Deploy*.

| Step | What you do | Lecture link | Time |
|---|---|---|---|
| 1 | Look at the data. See that two equal groups look unequal on paper. | World → Data → Labels | 20 min |
| 2 | Show that hiding the `group` column does nothing (it leaks). | "fairness through unawareness" | 15 min |
| 3 | Look at the trained model, then move the approval line and watch the gap. | Threshold: "where you cut" | 25 min |
| 4 | Try four fixes in one table; see which works and which makes it worse. | "fairness is not a feature you add" | 20 min |
| 5 | Read the completed model card and datasheet; check their numbers against yours. | Deploy: "who acts on it" | 20 min |
| | Check `SUBMISSION.md` (already filled in), run `make_submission.py`, upload one zip. | | 15 min |

**If any step takes more than 40 minutes, stop and post in the channel.** That is a bug in
the material, not in you.

**Where the answers come from.** Every answer in `SUBMISSION.md` is read off a printout in the
notebook, and every question says which line, row or column to look at. You never need the slides,
the booklet, the model card or the datasheet to verify Steps 1 to 4; those two files are only used in Step 5.

## Setup (once)

```bash
bash setup.sh                    # macOS / Linux   (Windows: .\setup.ps1)
```
or by hand:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-core.txt                # numpy, pandas, scikit-learn, matplotlib, scipy
python setup_check.py                               # tells you what is missing; never crashes
jupyter notebook week2_lab.ipynb
```
`shap`, `lime` and `ipywidgets` are **not needed**. In Google Colab, upload the folder and open the notebook.

## The dataset

`data/lending_data.csv`, 8,000 loan applicants, 7 permitted features, `group`, and the outcome `repaid`.
Open it in Excel if you like. `data/lending_data.json` records how it was made, and `datasheet.md`
documents it. Two extra columns, `repaid_true` and `latent`, would not exist in real life; they are
there so you can *prove* the two groups are equal. Never use them as features.

## The files (you only open the first four)

| file | what it is | do I need it? |
|---|---|---|
| `week2_lab.ipynb` | **The lab.** Open this. | **yes** |
| `data/lending_data.csv` | The dataset. | **yes** |
| `SUBMISSION.md` | The answer sheet, **already filled in** with the reference answers. Check each number against your run. | **yes** |
| `model_card.md`, `datasheet.md` | Completed model card and datasheet for this model and data. | **yes** |
| `make_submission.py` | Builds the one zip you upload. | yes, at the end |
| `helpers.py` | Tiny wrappers so each step is one cell. | no |
| `datasets.py`, `fairness.py`, `explorer.py`, `mitigate.py`, `explain.py` | The code behind the steps. You run it; you never edit it. | no |
| `run_lab.py` | Runs everything at once (about 40 s). | no |
| `model_card_template_full.md`, `datasheet_template.md` | The full professional templates (blank). Optional. | no |

## What you should find (so you know you are on track)

- **Step 1.** Both groups have identical ability (about −0.02 and −0.01). On paper, B repays 0.44 vs A 0.55.
- **Step 2.** `group` can be guessed from the permitted columns at **99.6%**; `postcode_score` is the leak.
- **Step 3.** At threshold 0.5 the approval gap is **0.20**. The gap only reaches ~0.02 at thresholds that approve ~2% or ~99% of everyone.
- **Step 4.** Dropping the proxy makes it **worse** (0.27). Fixing the label gets it to **0.03** for almost no accuracy.

## Optional, only if you want more

The notebook ends with *Going further*: which mechanism did the damage (one cell), the impossibility
table, explaining one refused applicant, a simpler model, and the full templates.

## Grading

Labs are not graded. Submit the zip so we can see the work and fix anything that did not work for you.
