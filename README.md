# Week 1 Lab — orientation, and getting your tools working

**The Evolution of Machine Learning and AI** · SETU

Two hours: thirty minutes on the AI Evolution Timeline as a macro orientation,
then environment setup for the remaining eleven weeks.

## Setup

```bash
bash setup.sh                    # macOS / Linux   (Windows: .\setup.ps1)
```

That creates the virtual environment, installs the **core** packages (small and
fast), and runs the environment check. When you have time and disk space before
Week 3, add the heavy stack:

```bash
bash setup.sh --full             # adds PyTorch, transformers, datasets, hf
```

Doing it by hand instead:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_check.py            # never crashes; tells you what is missing
```

**`INSTALL.md` is the full guide** — Python per operating system, virtual
environments, Git and GitHub authentication, Colab, HuggingFace tokens and the
`hf` CLI, Ollama, and a troubleshooting table for the eight errors that actually
come up.

`setup_check.py` is the first thing to run and the thing to bring to the lab if
something is wrong. It reports on Python, the scientific stack, PyTorch and any
accelerator, HuggingFace, Ollama, git identity, and whether you are in Colab.

## The files

| file | what it is |
|---|---|
| `timeline_data.py` | 62 milestones, 1843–2025, each tagged with its **family** (symbolic / connectionist / statistical) and the **ingredient** it supplied. The dataset, not a picture of one. |
| `ai_timeline.py` | Four figures: the hype wave, the pendulum counted, the ingredient supply chart, and the era bands. Plus an ipywidgets year scrubber. |
| `three_ingredients.py` | The Week 1 thesis as a **runnable experiment**. No GPU, no download. |
| `setup_check.py` | Environment report card. |
| `hello_models.py` | First contact with HuggingFace and with Ollama. |
| `week1_lab.ipynb` | The guided version. |

## The 30-minute orientation

```bash
python ai_timeline.py        # four figures into ./figures
python three_ingredients.py  # the experiment, about 60 seconds
```

`three_ingredients.py` sets **data**, **architecture** and **compute** to LOW or
HIGH independently and reports held-out accuracy for all eight combinations, on
sklearn's bundled digits. On the default seeds you should get roughly:

| combination | era it corresponds to | test accuracy |
|---|---|---|
| architecture only | perceptron, 1958: right idea, starved | ~0.86 (train 1.00) |
| compute only | Deep Blue, 1997: power, no transfer | ~0.45 |
| data + compute | ImageNet 2010: an arena with no champion | ~0.67 |
| **all three** | **AlexNet, 2012** | **~0.97** |

The lecture asserts that 2012 was a coincidence of supply rather than a new
idea. This is that assertion, measured.

## Exercises

1. **Read the tags.** `load()` gives you the timeline as a DataFrame. Plot
   milestones per decade by family. When does connectionism overtake symbolic
   AI, and does the crossover match the story in the lecture?
2. **Argue with the supply dates.** `SUPPLY` in `timeline_data.py` says
   architecture arrived in 1986, compute in 2007 and data in 2009. Change them,
   defend your version in three sentences, and note what it does to the gap
   before AlexNet. *There is no correct answer; there are defensible ones.*
3. **Find the dormant field.** List every milestone that occurred inside a
   winter. What does that do to the claim that the field was dead?
4. **Break the experiment.** In `three_ingredients.py`, find a setting where
   *more compute makes test accuracy worse*. Explain it in one sentence using a
   term from Week 3.
5. **Environment.** Get `setup_check.py` to zero blocking items. Push this
   folder to a GitHub repository; that repository is where your Week 12 project
   will live.

## Assessment link

LO1 is examined in the Week 6 quiz. The timeline dataset is the fastest
revision tool in the module: if you can reconstruct the era table from
`timeline_data.py` without looking, Week 1 revision is done.
