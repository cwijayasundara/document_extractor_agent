"""Cache utilities for LlamaParse results.

This module provides utilities to cache and retrieve parsed document content,
avoiding redundant API calls to LlamaParse for files that have already been processed.
"""

import os
from pathlib import Path
from typing import Optional


def get_cache_dir() -> Path:
    """Get the cache directory path for parsed files.

    Returns:
        Path: Absolute path to the cache directory
    """
    cache_dir = Path(__file__).parent / "parsed"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_cache_filename(file_path: str) -> str:
    """Generate cache filename from original file path.

    Args:
        file_path: Path to the original document file

    Returns:
        str: Cache filename in format '{original_filename}.md'

    Example:
        >>> get_cache_filename("/path/to/docs/IMG_4693.JPEG")
        'IMG_4693.JPEG.md'
    """
    original_filename = Path(file_path).name
    return f"{original_filename}.md"


def get_cache_path(file_path: str) -> Path:
    """Get the full cache file path for a given document.

    Args:
        file_path: Path to the original document file

    Returns:
        Path: Full path to the cached markdown file
    """
    cache_dir = get_cache_dir()
    cache_filename = get_cache_filename(file_path)
    return cache_dir / cache_filename


def get_cached_parse(file_path: str) -> Optional[str]:
    """Check if parsed content exists in cache and load it.

    Args:
        file_path: Path to the original document file

    Returns:
        Optional[str]: Cached parsed content if exists, None otherwise
    """
    cache_path = get_cache_path(file_path)

    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            # If reading fails, return None to trigger fresh parse
            print(f"Warning: Failed to read cache file {cache_path}: {e}")
            return None

    return None


def save_parsed_content(file_path: str, content: str) -> Path:
    """Save parsed content to cache as markdown file.

    Args:
        file_path: Path to the original document file
        content: Parsed content to save

    Returns:
        Path: Path to the saved cache file

    Raises:
        IOError: If writing to cache fails
    """
    cache_path = get_cache_path(file_path)

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return cache_path
    except Exception as e:
        raise IOError(f"Failed to save parsed content to {cache_path}: {e}")


def clear_cache(file_path: Optional[str] = None) -> None:
    """Clear cached parsed content.

    Args:
        file_path: Optional path to specific file to clear.
                  If None, clears entire cache directory.
    """
    if file_path:
        # Clear specific file cache
        cache_path = get_cache_path(file_path)
        if cache_path.exists():
            cache_path.unlink()
    else:
        # Clear entire cache directory
        cache_dir = get_cache_dir()
        for cache_file in cache_dir.glob("*.md"):
            cache_file.unlink()
