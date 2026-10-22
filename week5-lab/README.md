# Week 5 Lab: Teaching a Machine a Law of Nature

**The Evolution of Machine Learning and AI · SETU**

> **You do not need the booklet for this lab.** Every question says which printout to read.

The lecture had six parts. The lab has **six exercises, one per part, in the same order**. In each
one you **predict** before you run, **run** the experiment, **look** at the column the question
names, and **explain** the result in your own words. In Exercises 2 and 4 you also **build** the
three pieces that make a physics-informed network work: the residual, the physics loss and the hard
constraint. By the end of Exercise 4 everything except the network and the Week 3 training loop is
your own code, and it trains.

| Exercise | Lecture part | Slides | You build | Time |
|---|---|---|---|---|
| 1 · A law is not a pattern | Part 1 | 7 to 10 | | 20 min |
| 2 · Checking the equation where there is no data | Part 2 | 13 to 17 | the residual, the physics loss | 35 min |
| 3 · What the law buys | Part 3 | 21 to 24 | | 20 min |
| 4 · What it costs, and how it goes wrong | Part 4 | 27 to 34 | the hard constraint | 35 min |
| 5 · Four more ways to build in science | Part 5 | 37 to 43 | | 25 min |
| 6 · Choosing | Part 6 | 46 to 49 | | 20 min |
| | Two last questions, then upload `SUBMISSION.md` | | | 10 min |

About 2 h 45 min, or about 2 h 25 min if you skip the steps marked *optional*.
**If any exercise takes more than 40 minutes, stop and post in the channel.** That is a bug in
the material, not in you.

## What you need

Python 3.10 or newer with the packages in `requirements.txt`. **No PyTorch and no deep-learning
library**: the whole lab runs on numpy plus a small automatic-differentiation engine that ships
in the folder. Or Google Colab: upload the folder, open the notebook.

Cells that train are slower than other weeks (20 to 90 seconds each) because checking an equation
means taking derivatives of the network inside every training step. That is normal.

```bash
pip install -r requirements.txt      # numpy, pandas, matplotlib, notebook
jupyter notebook week5_lab.ipynb
```

## The files (you only open three)

| file | what it is | do I need it? |
|---|---|---|
| `week5_lab.ipynb` | **The lab.** Open this. | **yes** |
| `SUBMISSION.md` | **The answer sheet, blank.** Fill it in as you go, add your name, upload it. | **yes** |
| `WHERE_TO_FIND.md` | One table: for each question, which cell to run and which printout to read. | **yes** |
| `lecture_steps.py` | One helper per lecture idea, and the checks for the code you write. You never edit it. | no |
| `autodiff.py`, `mlp.py`, `pinn.py`, `heat.py`, `symmetry.py`, `neural_ode.py`, `operators.py`, `diffusion.py` | The code behind the exercises, every derivative written out. | no |
| `run_lab.py` | The full run that reproduces every number in the lecture (several minutes). Optional. | no |
| `interactive/*.html` | Three browser toys. Double-click; no install. | optional, fun |

## The reference answers

They are a separate card on Tutors, **Week 5 Reference Answers**. Open it only after you have
uploaded `SUBMISSION.md`, and compare it with what you wrote.

## Grading

Labs are not graded. Upload `SUBMISSION.md` so we can see the work and fix anything that did not work for you.
