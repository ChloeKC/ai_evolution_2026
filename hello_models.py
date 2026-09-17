"""
hello_models.py — first contact with a hosted model and a local one.

The Evolution of Machine Learning and AI · Week 1 Lab
South East Technological University

Two runtimes you will use for the rest of the module:

    HuggingFace  the library and hub Weeks 3-9 pull models from
    Ollama       the local runtime for Weeks 7 and 9, where the lean turn
                 stops being a slide and becomes a thing on your laptop

Everything here is network-gated and fails politely. Run it before Week 3, not
during it: the first HuggingFace download is a few hundred megabytes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"


def hf_hello(model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """Smallest useful HuggingFace pipeline: sentiment on two sentences."""
    try:
        from transformers import pipeline
    except ImportError:
        print("transformers not installed — pip install transformers torch")
        return None
    try:
        clf = pipeline("sentiment-analysis", model=model)
    except Exception as e:
        print(f"could not load {model}: {type(e).__name__}: {e}")
        print("usually a network or disk-space problem; try again on the campus network")
        return None

    tests = [
        "The perceptron was the embryo of a conscious machine.",
        "It could see four hundred pixels.",
    ]
    for t in tests:
        r = clf(t)[0]
        print(f"  {r['label']:8s} {r['score']:.3f}   {t}")
    print("\n  Week 1 question: is this understanding, or fluency? (booklet Ch. 5)")
    return clf


def ollama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:
        return []


def ollama_hello(model: str | None = None, prompt: str | None = None):
    """One local generation, with no API key and no data leaving the machine."""
    if not ollama_running():
        print("Ollama is not responding on localhost:11434.")
        print("  install:  https://ollama.com")
        print("  start:    ollama serve")
        print("  a model:  ollama pull llama3.2:1b     (about 1.3 GB)")
        return None

    models = ollama_models()
    if not models:
        print("Ollama is running but has no models. Try: ollama pull llama3.2:1b")
        return None
    model = model or models[0]
    prompt = prompt or ("In two sentences, what was the AI winter of the 1970s, "
                        "and what caused it?")

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
    except Exception as e:
        print(f"generation failed: {type(e).__name__}: {e}")
        return None

    print(f"  model: {model}\n")
    print(out.get("response", "").strip())
    print("\n  This ran on your machine. No API key, no data sent anywhere.")
    print("  Week 9 is about why that is now possible at all.")
    return out.get("response")


if __name__ == "__main__":
    print("=" * 72); print("1 · HuggingFace (hosted weights, local inference)"); print("=" * 72)
    hf_hello()
    print("\n" + "=" * 72); print("2 · Ollama (fully local)"); print("=" * 72)
    ollama_hello()
