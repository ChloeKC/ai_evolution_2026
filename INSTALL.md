# Getting set up — Python, Colab, GitHub, HuggingFace, Ollama

**The Evolution of Machine Learning and AI** · SETU · Week 1 Lab

Budget about 45 minutes. Do it once, and the remaining eleven weeks just work.
If anything here fails, run `python setup_check.py` and bring its output to the lab.

| Tool | What it is for | First needed |
|---|---|---|
| **Python 3.11+** | The language everything else runs on | Week 1 |
| **A virtual environment** | Keeps this module's packages away from everything else on your machine | Week 1 |
| **Git + GitHub** | Where every deliverable in this module lives, including the Week 12 project | Week 1 |
| **Google Colab** | A free GPU when your laptop is not enough | Weeks 7–9 |
| **HuggingFace** | Where the models and datasets come from | Week 3 |
| **Ollama** | Runs models entirely on your own machine, no API key | Weeks 7 and 9 |

---

## 1 · Python

You need **3.11 or newer**. Check what you have:

```bash
python3 --version        # macOS / Linux
python --version         # Windows
```

If that fails or shows 3.10 or older:

- **Windows** — install from the Microsoft Store ("Python 3.12"), or python.org.
  On the python.org installer, **tick "Add python.exe to PATH"** on the first screen.
  Getting this wrong is the single most common cause of "python is not recognised".
- **macOS** — `brew install python@3.12`, or the python.org installer.
  Do **not** rely on the `/usr/bin/python3` that ships with macOS.
- **Linux** — `sudo apt install python3 python3-venv python3-pip` (Debian/Ubuntu).
  The `python3-venv` package is separate and is easy to miss.

### The rule that saves you a week of pain

> **Never `pip install` into your system Python.** Always work inside a virtual
> environment. When two modules need different versions of the same library —
> and they will — the venv is what stops them fighting.

---

## 2 · The virtual environment

From inside this folder:

```bash
python3 -m venv .venv                 # create it (once)

source .venv/bin/activate             # activate: macOS / Linux
.venv\Scripts\activate                # activate: Windows PowerShell

pip install -r requirements.txt       # install this module's packages
```

Your prompt should now start with `(.venv)`. That prefix is the whole point: it
tells you which Python you are about to run. If it disappears (new terminal
window, restarted editor), **activate again** — you have not broken anything.

To leave: `deactivate`.

There is a one-command version in `setup.sh` (macOS/Linux) and `setup.ps1`
(Windows) that does all of the above and then runs the environment check.

### If you prefer `uv`

`uv` is a much faster drop-in replacement and is fine to use here:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS / Linux
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 3 · Verify

```bash
  python setup_check.py
```

It never crashes. It prints one line per tool and ends with a count of
**blocking** items. Aim for zero blocking; the "sort out later" items are things
you do not need until Week 3 or Week 7.

---

## 4 · Git and GitHub

Your GitHub repository **is** the module deliverable. Every week adds to it, and
the Week 12 project is submitted from it.

```bash
git --version                         # install from git-scm.com if missing

git config --global user.name  "Your Name"
git config --global user.email "you@setu.ie"      # use your student address
```

Then, on github.com: **New repository** → name it something like
`ai-evolution-2026` → **Private** → Create. Follow the "push an existing
repository" instructions it shows you, or:

```bash
git init
git add .
git commit -m "Week 1: environment and timeline lab"
git branch -M main
git remote add origin https://github.com/YOURNAME/ai-evolution-2026.git
git push -u origin main
```

**Authentication.** GitHub stopped accepting passwords over HTTPS in 2021. Use
either:

- a **personal access token** (Settings → Developer settings → Tokens) pasted
  when git asks for a password, or
- the **GitHub CLI**: `gh auth login`, which handles it for you, or
- **SSH keys**: `ssh-keygen -t ed25519 -C "you@setu.ie"`, then paste
  `~/.ssh/id_ed25519.pub` into Settings → SSH keys.

A `.gitignore` is included in this folder. It keeps `.venv/`, caches, model
weights and `.env` out of your repository. **Never commit a token.**

---

## 5 · Google Colab

Nothing to install: it is a notebook in your browser with a free GPU attached.
Go to [colab.research.google.com](https://colab.research.google.com) and sign in
with a Google account.

- **Runtime → Change runtime type → T4 GPU** to get an accelerator.
- Colab resets when it disconnects. Anything you want to keep must be pushed to
  GitHub or saved to Drive.
- To run this lab there, put this in the first cell:

```python
!git clone https://github.com/YOURNAME/ai-evolution-2026.git
%cd ai-evolution-2026/week1-lab
!pip install -q -r requirements.txt
```

Use Colab when your own machine cannot cope — realistically Weeks 7 to 9. For
Weeks 1 to 5, a laptop is entirely sufficient.

---

## 6 · HuggingFace

The Hub is where the models and datasets in Weeks 3 to 11 come from.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Settings → **Access Tokens** → New token → **Read** access is enough for now.
3. Log in from your machine:

```bash
pip install -U huggingface_hub
hf auth login              # paste the token when prompted
hf auth whoami             # confirms who you are
hf version
```

> The CLI used to be called `huggingface-cli`. It was renamed to `hf` in 2025.
> The old name still works but prints a deprecation warning; tutorials you find
> online will often still use it. `huggingface-cli login` == `hf auth login`.

Your first model:

```bash
python hello_models.py
```

**Where downloads go.** `~/.cache/huggingface` by default, and it gets large.
`hf cache scan` shows what is there; `hf cache delete` reclaims space. Set
`HF_HOME` to move it to another drive if your laptop is tight.

**Gated models.** Some (Llama, for instance) require you to accept a licence on
the model page while logged in. You will hit this in Week 7, not before.

---

## 7 · Ollama

Ollama runs models **entirely on your own machine**: no API key, no account, and
no data leaving the laptop. It is the runtime behind Weeks 7 and 9, and the
single most persuasive demonstration of the lean turn in the whole module.

- **macOS / Windows** — download the installer from [ollama.com](https://ollama.com).
- **Linux** — `curl -fsSL https://ollama.com/install.sh | sh`

Then:

```bash
ollama --version
ollama serve                      # starts the background service if not running
ollama pull llama3.2:1b           # about 1.3 GB — start small
ollama run llama3.2:1b "In two sentences, what was the AI winter of the 1970s?"
ollama list                       # what you have downloaded
```

It also exposes an HTTP API on `localhost:11434`, which is what `hello_models.py`
uses and what your Week 10 retrieval application will talk to:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Why did AlexNet matter?",
  "stream": false
}'
```

**Sizing.** A 1B model needs roughly 2 GB of RAM; 7–8B models need about 8 GB;
larger ones want a GPU. Start with `llama3.2:1b` and only go bigger if it is
comfortable.

---

## 8 · When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `python: command not found` | Not on PATH, or it is `python3` on your system | Try `python3`. On Windows, reinstall ticking "Add to PATH" |
| `pip: externally-managed-environment` | You are installing into system Python | Activate the venv first. This error is the venv rule being enforced |
| `(.venv)` vanished from the prompt | New terminal window | Activate again; nothing is broken |
| `ModuleNotFoundError` after installing | Editor is using a different interpreter | Point VS Code at `.venv/bin/python` (Command Palette → Python: Select Interpreter) |
| `remote: Support for password authentication was removed` | Git wants a token, not a password | Use a personal access token, `gh auth login`, or SSH |
| `401 Unauthorized` from HuggingFace | Not logged in, or gated model | `hf auth login`, and accept the licence on the model page |
| Ollama: `connection refused` on 11434 | The service is not running | `ollama serve` in a separate terminal |
| Colab: "cannot connect to GPU backend" | Free tier is busy or your quota is used | Wait, or continue on CPU — Weeks 1–5 do not need a GPU |

---

## Before next week

- [ ] `python setup_check.py` reports **0 blocking**
- [ ] A private GitHub repository exists, and this folder is pushed to it
- [ ] `hf auth whoami` returns your username
- [ ] `ollama run llama3.2:1b "hello"` answers on your own machine
- [ ] You have opened one Colab notebook and seen it run a cell
