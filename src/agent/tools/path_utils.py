"""Utilities for working with document file paths in agent tools."""

from pathlib import Path
from typing import Iterable, List


def _get_repo_root() -> Path:
    """Return the repository root based on this module location."""
    return Path(__file__).resolve().parents[3]


def _dedupe_candidates(candidates: Iterable[Path]) -> List[Path]:
    """Return candidates with duplicates removed while preserving order."""
    unique_candidates: List[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    return unique_candidates


def resolve_document_path(file_path: str) -> Path:
    """Resolve a document path against common project locations.

    Args:
        file_path: User supplied path to the document.

    Returns:
        Path: Absolute path to an existing document file.

    Raises:
        FileNotFoundError: If the document cannot be located.
        ValueError: If the provided path is empty.
    """
    if not file_path or not file_path.strip():
        raise ValueError("Document path is empty. Provide a valid PDF or image file path.")

    raw_path = Path(file_path).expanduser()
    repo_root = _get_repo_root()

    candidate_paths = []

    # Always check the raw path first (absolute or relative to current working directory)
    candidate_paths.append(raw_path if raw_path.is_absolute() else Path.cwd() / raw_path)

    if not raw_path.is_absolute():
        candidate_paths.append(repo_root / raw_path)

    # Fall back to well-known document locations when only a filename was provided
    uploads_dir = repo_root / "uploads"
    docs_dir = repo_root / "docs"

    candidate_paths.append(uploads_dir / raw_path.name)
    candidate_paths.append(docs_dir / raw_path.name)

    for candidate in _dedupe_candidates(candidate_paths):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    attempted_paths = [str(path) for path in _dedupe_candidates(candidate_paths)]
    attempted_locations = "\n".join(f"- {path}" for path in attempted_paths)
    raise FileNotFoundError(
        f"Document file not found for '{file_path}'. Checked the following locations:\n{attempted_locations}"
    )
