# Week 3 Lab: How Machines Learn

**The Evolution of Machine Learning and AI · SETU**

> **You do not need the booklet for this lab.** Every answer is a printout in the notebook,
> and every question says where to look.

The lecture had six parts. The lab has **six exercises, one per part, in the same order**.
Each does on your machine exactly what the matching slides did on the screen. About 20 minutes each.

| Exercise | Lecture part | Slides | Time |
|---|---|---|---|
| 1 · A formula with dials | Part 1, what learning means | 6 to 11 | 20 min |
| 2 · Four shapes of model | Part 2 (Live demo 1) | 12 to 19 | 20 min |
| 3 · Memorising is not learning | Part 3 | 20 to 23 | 20 min |
| 4 · Walking downhill | Part 4 (Live demo 2) | 24 to 30 | 20 min |
| 5 · The neuron, the wall, the hidden layer | Part 5 (Live demo 3) | 31 to 37 | 20 min |
| 6 · Blame, and the five lines | Part 6 | 38 to 43 | 20 min |
| | Check `SUBMISSION.md` (already filled in) against your run and upload it | | 10 min |

**If any exercise takes more than 40 minutes, stop and post in the channel.** That is a bug in
the material, not in you.

## What you need

Python 3.10 or newer with the packages in `requirements.txt`. **You do not need PyTorch**
(Exercise 6 prints the five lines without it). Or Google Colab: upload the folder, open the notebook.

```bash
pip install -r requirements.txt      # numpy, pandas, scikit-learn, matplotlib, notebook. About one minute.
jupyter notebook week3_lab.ipynb
```

## The files (you only open three)

| file | what it is | do I need it? |
|---|---|---|
| `week3_lab.ipynb` | **The lab.** Open this. | **yes** |
| `SUBMISSION.md` | The answer sheet, already filled in. Check it against your run, add your name, upload it. | **yes** |
| `WHERE_TO_FIND.md` | One table: what to run, what prints, where each answer is. | **yes** |
| `lecture_steps.py` | One helper per lecture part. You run them; you never edit it. | no |
| `classical.py`, `gradient_descent.py`, `scratch_net.py`, `torch_intro.py` | The code behind the exercises. | no |
| `interactive/*.html` | The two browser toys from Live demos 2 and 3. Double-click; no install. | optional, fun |

## Grading

Labs are not graded. Upload `SUBMISSION.md` so we can see the work and fix anything that did not work for you.
