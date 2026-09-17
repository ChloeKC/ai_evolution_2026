#!/usr/bin/env python3
"""
make_submission.py — package this week's work into one file to upload.

Run this at the end of every lab:

    python make_submission.py

It checks your work, bundles it, and produces a single .zip file for you to
upload. That is your submission. Nothing else is required.

This script uses only the Python standard library, never installs anything,
and never sends anything anywhere. Everything it does happens on your machine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAVED = HERE / ".student_number"

# Files gathered into the bundle, relative to this folder.
INCLUDE_NAMES = ["SUBMISSION.md"]
INCLUDE_GLOBS = ["*.png", "*.jpg", "*.jpeg"]

# Files that ship with the lab. Never bundled from the folder root, even if
# they match a glob above — they are mine, not the student's work.
LAB_FILES = {
    "requirements.txt", "requirements-full.txt", "setup.sh", "setup.ps1",
    "setup_check.py", "make_submission.py", "INSTALL.md", "README.md",
    "timeline_data.py", "ai_timeline.py", "three_ingredients.py",
    "hello_models.py", "week1_lab.ipynb",
}
INCLUDE_DIRS = ["submission"]

# Never bundle these, even if they match above.
SKIP_DIR_PARTS = {".venv", "venv", "__pycache__", ".git", "figures",
                  ".ipynb_checkpoints", "node_modules"}

BAR = "-" * 66


def say(msg=""):
    print(msg, flush=True)


def bail(msg, hint=""):
    say()
    say(BAR)
    say(f"  STOPPED: {msg}")
    if hint:
        say()
        for line in hint.strip().splitlines():
            say(f"  {line}")
    say(BAR)
    say()
    sys.exit(1)


# --------------------------------------------------------------------------
# Which week is this?
# --------------------------------------------------------------------------

def detect_week() -> int:
    """week1-lab -> 1. Falls back to asking."""
    m = re.search(r"week[-_ ]?(\d+)", HERE.name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"week[-_ ]?(\d+)", HERE.parent.name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


# --------------------------------------------------------------------------
# Who are you?
# --------------------------------------------------------------------------

def student_from_submission_md() -> str:
    """Read '**Student number:** 20012345' out of SUBMISSION.md."""
    path = HERE / "SUBMISSION.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\*\*Student number:\*\*\s*([A-Za-z0-9._-]{4,20})", text)
    return m.group(1).strip() if m else ""


def resolve_student(arg: str | None) -> str:
    from_md = student_from_submission_md()

    if arg:
        num = arg.strip()
    elif from_md:
        num = from_md
        say(f"Student number: {num}   (read from SUBMISSION.md)")
    elif SAVED.exists():
        num = SAVED.read_text(encoding="utf-8").strip()
        say(f"Student number: {num}   (remembered — delete "
            f".student_number to change it)")
    else:
        say()
        say("Your student number is used to name the file you upload,")
        say("so I know whose work it is.")
        say()
        try:
            num = input("  Student number: ").strip()
        except (EOFError, KeyboardInterrupt):
            bail("No student number given.")

    if not num:
        bail("No student number given.",
             "Run again and type your student number when asked, or:\n"
             "    python make_submission.py --student 20012345")

    if not re.fullmatch(r"[A-Za-z0-9._-]{4,20}", num):
        bail(f"'{num}' does not look like a student number.",
             "Use the number from your student card — digits only,\n"
             "no spaces and no name.")

    if not SAVED.exists():
        try:
            SAVED.write_text(num + "\n", encoding="utf-8")
        except OSError:
            pass

    return num


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_submission_md(force: bool) -> str:
    path = HERE / "SUBMISSION.md"
    if not path.exists():
        bail("SUBMISSION.md is missing from this folder.",
             "It should have come with the lab. Re-download the lab archive\n"
             "from the course page and copy SUBMISSION.md across.")

    text = path.read_text(encoding="utf-8", errors="replace")
    problems = []

    if "TODO" in text:
        n = text.count("TODO")
        problems.append(f"{n} TODO marker(s) still in SUBMISSION.md")
    if re.search(r"^- \[ \]", text, re.MULTILINE):
        n = len(re.findall(r"^- \[ \]", text, re.MULTILINE))
        problems.append(f"{n} unticked checklist item(s)")
    if "YOUR-USERNAME" in text or "YOUR NAME" in text.upper():
        problems.append("placeholder text not replaced")

    if problems and not force:
        bail("SUBMISSION.md is not finished.",
             "\n".join(f"- {p}" for p in problems) + "\n\n"
             "Open SUBMISSION.md, answer the questions, delete the TODO\n"
             "markers, then run this again.\n\n"
             "If you deliberately want to submit it incomplete:\n"
             "    python make_submission.py --force")

    for p in problems:
        say(f"  ! {p}")

    return text


def run_setup_check() -> tuple[str, int | None]:
    """Run setup_check.py and pull out the blocking count. Never raises."""
    script = HERE / "setup_check.py"
    if not script.exists():
        return "setup_check.py not found in this folder.", None

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(HERE), capture_output=True, text=True, timeout=300,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return "setup_check.py timed out after 5 minutes.", None
    except Exception as e:                                  # noqa: BLE001
        return f"Could not run setup_check.py: {e}", None

    m = re.search(r"(\d+)\s+blocking", output)
    blocking = int(m.group(1)) if m else None
    return output.strip(), blocking


# --------------------------------------------------------------------------
# Gathering
# --------------------------------------------------------------------------

def gather() -> list[Path]:
    found: list[Path] = []

    for name in INCLUDE_NAMES:
        p = HERE / name
        if p.is_file():
            found.append(p)

    for pattern in INCLUDE_GLOBS:
        for p in sorted(HERE.glob(pattern)):
            if p.is_file() and p.name not in LAB_FILES and p not in found:
                found.append(p)

    for dirname in INCLUDE_DIRS:
        d = HERE / dirname
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if SKIP_DIR_PARTS & set(p.relative_to(HERE).parts):
                continue
            if p.stat().st_size > 20 * 1024 * 1024:
                say(f"  ! skipping {p.name} — larger than 20 MB")
                continue
            found.append(p)

    return found


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Package this week's lab work into one file to upload.")
    ap.add_argument("--student", help="your student number")
    ap.add_argument("--week", type=int, help="week number (usually detected)")
    ap.add_argument("--force", action="store_true",
                    help="bundle even if SUBMISSION.md is unfinished")
    args = ap.parse_args()

    say()
    say(BAR)
    say("  BUILDING YOUR SUBMISSION")
    say(BAR)

    week = args.week or detect_week()
    if not week:
        bail("Could not work out which week this is.",
             "Run it again with the week number:\n"
             "    python make_submission.py --week 1")

    student = resolve_student(args.student)

    say()
    say(f"  Week {week} · student {student}")
    say()

    say("  Checking SUBMISSION.md ...")
    check_submission_md(args.force)
    say("  ok")

    say("  Running setup_check.py ...")
    check_output, blocking = run_setup_check()
    if blocking is None:
        say("  ! could not read a blocking count — bundling anyway")
    elif blocking > 0:
        say(f"  ! {blocking} blocking item(s). Your environment is not ready.")
        say("    You can still submit; fix them before next week.")
    else:
        say("  ok — 0 blocking")

    say("  Gathering files ...")
    files = gather()
    for p in files:
        say(f"    + {p.relative_to(HERE)}")
    if not files:
        bail("Nothing to bundle.",
             "SUBMISSION.md should at least be here. Re-download the lab.")

    manifest = {
        "student_number": student,
        "week": week,
        "built_at_utc": dt.datetime.now(dt.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "built_at_local": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lab_folder": HERE.name,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "setup_check_blocking": blocking,
        "submission_forced": bool(args.force),
        "files": [
            {"path": str(p.relative_to(HERE)).replace("\\", "/"),
             "bytes": p.stat().st_size,
             "sha256_16": sha256(p)}
            for p in files
        ],
    }

    out_name = f"submission-week{week:02d}-{student}.zip"
    out_path = HERE.parent / out_name

    try:
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, str(p.relative_to(HERE)).replace("\\", "/"))
            z.writestr("setup_check_output.txt", check_output)
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
    except OSError as e:
        bail(f"Could not write the zip file: {e}",
             "Check you have space and that the folder is not read-only.")

    size_kb = out_path.stat().st_size / 1024

    say()
    say(BAR)
    say("  DONE")
    say(BAR)
    say()
    say(f"  File:  {out_name}")
    say(f"  Where: {out_path.parent}")
    say(f"  Size:  {size_kb:.0f} KB")
    say()
    say("  NEXT STEP — upload that one file. Nothing else.")
    say()
    say("  Upload link is in the 'Submitting This Lab' page of the lab book.")
    say()
    if blocking:
        say(f"  Note: {blocking} blocking environment item(s). Ask in the")
        say("  module channel before next week's lab.")
        say()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nCancelled. Nothing was written.")
        sys.exit(1)
