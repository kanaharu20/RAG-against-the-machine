"""Loading of the question datasets.

The dataset files are JSON objects holding a single ``rag_questions``
array, so they map onto :class:`RagDataset` directly. Both flavours go
through the same loader: the union in the model tells an answered
question from an unanswered one by the fields it carries.

Every failure is reported as :class:`DatasetError` so that the CLI has
a single exception to catch. The subject requires missing files and
malformed JSON to be handled gracefully rather than crashing.
"""

from pathlib import Path

from pydantic import ValidationError

from .models import RagDataset


class DatasetError(Exception):
    """A dataset file could not be read or did not validate."""


def load_dataset(path: str | Path) -> RagDataset:
    """Read a dataset file into a validated :class:`RagDataset`.

    Args:
        path: Path to a dataset JSON file.

    Returns:
        The parsed dataset.

    Raises:
        DatasetError: If the file is unreadable, is not valid JSON, or
            does not match the expected schema.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatasetError(f"cannot read {file_path}: {error}") from error
    try:
        return RagDataset.model_validate_json(raw)
    except ValidationError as error:
        raise DatasetError(
            f"invalid dataset {file_path}: {error}"
        ) from error
