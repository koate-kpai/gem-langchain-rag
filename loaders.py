# loaders.py
"""
Extensible document loader registry.

Design Pattern: Registry (aka Plugin Pattern)
------------------------------------------------
Rather than hardcoding a single loader, this module maintains a mapping of
file extensions → LangChain loader classes. Adding a new format requires
only registering it in the LOADER_REGISTRY dict — no changes to ingest.py.

Supported formats:
  - .txt  → TextLoader (built-in LangChain, zero dependencies)
  - .md   → TextLoader (Markdown is plaintext-compatible)
  - .pdf  → PyPDFLoader (requires `pip install pypdf`)

Cost rationale: validating file existence, size, and encoding BEFORE invoking
a loader prevents wasted downstream compute (API calls to OpenAI for embedding
empty or corrupted documents costs real money).
"""
import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loader registry: extension → (loader_class, description)
# ---------------------------------------------------------------------------
# To add a new format (e.g. .docx), install the required package and add:
#   from langchain_community.document_loaders import Docx2txtLoader
#   ".docx": (Docx2txtLoader, "Word Document"),
#
# The registry pattern eliminates if/elif chains and makes format support
# discoverable in a single location — a textbook Open/Closed Principle
# implementation (open for extension, closed for modification).
LOADER_REGISTRY: dict[str, tuple] = {
    ".txt": (TextLoader, "Plain Text"),
    ".md": (TextLoader, "Markdown"),
}


try:
    from langchain_community.document_loaders import PyPDFLoader
    LOADER_REGISTRY[".pdf"] = (PyPDFLoader, "PDF Document")
    logger.info("PDF support enabled (pypdf is installed).")
except ImportError:
    logger.info(
        "PDF support not available. Install with: pip install pypdf"
    )


# Labels that are NOT text-based and should be skipped with a warning
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".zip", ".exe", ".bin"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_loader_for_file(file_path: Path):
    """Return the appropriate LangChain loader for a given file.

    Args:
        file_path: Path to the source document.

    Returns:
        An instantiated LangChain document loader.

    Raises:
        ValueError: If the file extension is unsupported or binary.
    """
    ext = file_path.suffix.lower()

    if ext in BINARY_EXTENSIONS:
        raise ValueError(
            f"Skipping '{file_path.name}': binary format '{ext}' is not supported. "
            "Only text-based documents (.txt, .md, .pdf) can be ingested."
        )
    if ext not in LOADER_REGISTRY:
        raise ValueError(
            f"Unsupported file format '{ext}' for '{file_path.name}'. "
            f"Supported formats: {', '.join(sorted(LOADER_REGISTRY.keys()))}"
        )

    loader_class, description = LOADER_REGISTRY[ext]
    logger.debug("Selected loader '%s' for %s (%s)", loader_class.__name__, file_path.name, description)
    return loader_class(str(file_path))


def find_documents(data_dir: Path, glob_pattern: str = "*.*") -> list[Path]:
    """Glob the data directory for supported documents, sorted alphabetically.

    Skips unsupported and binary extensions with a warning.

    Args:
        data_dir: Directory to scan for documents.
        glob_pattern: Glob pattern (e.g. "*.*", "*.txt", "policy.*").

    Returns:
        Sorted list of file paths with supported extensions.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates = sorted(data_dir.glob(glob_pattern))
    valid: list[Path] = []

    for path in candidates:
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in LOADER_REGISTRY:
            valid.append(path)
        elif ext in BINARY_EXTENSIONS:
            logger.warning("Skipping binary file '%s' (%s)", path.name, ext)
        else:
            logger.warning(
                "Skipping '%s' — unsupported format '%s'. "
                "Supported: %s",
                path.name,
                ext or "(no extension)",
                ", ".join(sorted(LOADER_REGISTRY.keys())),
            )

    if not valid:
        logger.warning(
            "No supported documents found in '%s'. "
            "Add .txt, .md, or .pdf files and re-run.",
            data_dir,
        )

    return valid


def validate_file_size(file_path: Path, max_size_mb: int = 50) -> None:
    """Validate that a file does not exceed the maximum size.

    Args:
        file_path: Path to the file to check.
        max_size_mb: Maximum allowed size in megabytes.

    Raises:
        ValueError: If the file exceeds the size limit.
    """
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(
            f"'{file_path.name}' is {size_mb:.1f} MB — exceeds the "
            f"{max_size_mb} MB limit. Increase config.ingestion.max_file_size_mb "
            "if you need to process larger files."
        )
