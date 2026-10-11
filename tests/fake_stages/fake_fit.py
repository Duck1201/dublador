"""Fake `fit` for the fit loop (see fake_loop.py)."""

from fake_loop import fit
from stagekit import main

if __name__ == "__main__":
    main("fit", run=fit)
