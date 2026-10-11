"""Fake `translate` for the fit loop (see fake_loop.py)."""

from fake_loop import translate
from stagekit import main

if __name__ == "__main__":
    main("translate", run=translate)
