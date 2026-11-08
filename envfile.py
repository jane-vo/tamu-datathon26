"""
envfile.py - load KEY=VALUE lines from the .env file in the repo root into os.environ.

No extra package needed. Variables already set in your shell win over the file.
Lines starting with # are comments; values may be wrapped in "double" or 'single' quotes.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(path=None):
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and value and key not in os.environ:
                os.environ[key] = value
