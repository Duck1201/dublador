"""File contract shared by the dublador orchestrator and its stages."""

from stagekit.fileio import atomic_path, atomic_write_bytes, atomic_write_json, atomic_write_text
from stagekit.hashing import hash_file, hash_value
from stagekit.progress import Progress
from stagekit.schemas import ContractError, read_json, validate, write_json
from stagekit.stage import STAGE_DIRS, StageContext, StageError, load_config, main

__all__ = [
    "STAGE_DIRS",
    "ContractError",
    "Progress",
    "StageContext",
    "StageError",
    "atomic_path",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "hash_file",
    "hash_value",
    "load_config",
    "main",
    "read_json",
    "validate",
    "write_json",
]
