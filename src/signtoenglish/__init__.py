"""SignToEnglish: continuous ASL-to-English translation demo."""

import os

# Model hubs are offline unless a process opts in explicitly (CLAUDE.md, stop point 2):
# nothing may be fetched implicitly by `transformers` or `huggingface_hub`. These must be
# set before those libraries are imported, which is why they live in the package root.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.1.0"
