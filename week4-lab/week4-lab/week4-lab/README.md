# Week 4 Lab: Three Ways to Wire a Machine

**The Evolution of Machine Learning and AI · SETU**

> **You do not need the booklet for this lab.** Every answer is a printout in the notebook,
> and every question says where to look.

The lecture had six parts. The lab has **six exercises, one per part, in the same order**.
Each does on your machine exactly what the matching slides did on the screen. About 20 minutes each.

| Exercise | Lecture part | Slides | Time |
|---|---|---|---|
| 1 · A guess about the world | Part 1 | 7 to 13 | 20 min |
| 2 · Convolution: things near each other | Part 2 | 14 to 20 | 20 min |
| 3 · What the guess buys, and when it does not pay | Part 3 | 21 to 25 | 20 min |
| 4 · Recurrence: one thing after another | Part 4 | 26 to 35 | 20 min |
| 5 · Attention: who should talk to whom | Part 5 | 36 to 45 | 20 min |
| 6 · The bargain | Part 6 | 46 to 50 | 20 min |
| | Check `SUBMISSION.md` (already filled in) against your run and upload it | | 10 min |

**If any exercise takes more than 40 minutes, stop and post in the channel.** That is a bug in
the material, not in you.

## What you need

Python 3.10 or newer with the packages in `requirements.txt`. **You do not need PyTorch.**
Or Google Colab: upload the folder, open the notebook. A few cells train small networks and take
20 to 60 seconds; that is normal.

```bash
pip install -r requirements.txt      # numpy, pandas, scikit-learn, matplotlib, notebook
jupyter notebook week4_lab.ipynb
```

## The files (you only open three)

| file | what it is | do I need it? |
|---|---|---|
| `week4_lab.ipynb` | **The lab.** Open this. | **yes** |
| `SUBMISSION.md` | The answer sheet, already filled in. Check it against your run, add your name, upload it. | **yes** |
| `WHERE_TO_FIND.md` | One table: what to run, what prints, where each answer is. | **yes** |
| `lecture_steps.py` | One helper per lecture idea. You run them; you never edit it. | no |
| `convolution.py`, `cnn_numpy.py`, `inductive_bias.py`, `sequences.py`, `attention.py`, `optim.py` | The code behind the exercises, every derivative written out. | no |
| `torch_models.py`, `run_lab.py` | The PyTorch versions and the full four-minute run. Optional. | no |
| `interactive/*.html` | Three browser toys. Double-click; no install. | optional, fun |

## Grading

Labs are not graded. Upload `SUBMISSION.md` so we can see the work and fix anything that did not work for you.
