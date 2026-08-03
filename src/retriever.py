"""Retrieval over a persisted index, and self-evaluation.

Everything here is plain library code: the CLI calls it, and so could
an HTTP server or a notebook. Keeping the logic out of the command
layer is what makes the pipeline reusable.

The evaluation mirrors the metric the grader uses, so that iterating
locally does not require running the reference executable. A reference
source counts as found when a retrieved source sits in the same file
and its character range overlaps by at least ``IOU_THRESHOLD``.
"""

from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from .indexer import BM25Index
from .load_json import DatasetError, summarize_validation_error
from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

DEFAULT_K = 10
IOU_THRESHOLD = 0.05


def search_dataset(
    index: BM25Index, dataset: RagDataset, k: int
) -> StudentSearchResults:
    """Run the index over every question of a dataset.

    Args:
        index: The loaded index.
        dataset: Questions to answer, answered or not.
        k: Number of sources to retrieve per question.

    Returns:
        The results, ready to be serialised.
    """
    results = [
        MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=index.search(question.question, k),
        )
        for question in tqdm(
            dataset.rag_questions, desc="Searching", unit="question"
        )
    ]
    return StudentSearchResults(search_results=results, k=k)


def save_search_results(
    results: StudentSearchResults,
    save_directory: str | Path,
    filename: str,
) -> Path:
    """Write results next to their dataset name, creating the folder.

    The public datasets share file names, so the caller is expected to
    scope ``save_directory`` by dataset to avoid overwriting a previous
    run.
    """
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        results.model_dump_json(indent=2), encoding="utf-8"
    )
    return path


def load_search_results(path: str | Path) -> StudentSearchResults:
    """Read back a results file.

    Raises:
        DatasetError: If the file is unreadable or does not validate.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatasetError(
            f"cannot read {file_path}: {error}"
        ) from error
    try:
        return StudentSearchResults.model_validate_json(raw)
    except ValidationError as error:
        raise DatasetError(
            f"invalid search results {file_path}: "
            f"{summarize_validation_error(error)}"
        ) from error


def _overlap_ratio(
    first: MinimalSource, second: MinimalSource
) -> float:
    """Intersection over union of two character ranges."""
    start = max(
        first.first_character_index, second.first_character_index
    )
    stop = min(
        first.last_character_index, second.last_character_index
    )
    intersection = max(0, stop - start)
    union = max(
        first.last_character_index, second.last_character_index
    ) - min(
        first.first_character_index, second.first_character_index
    )
    return intersection / union if union > 0 else 0.0


def _is_found(
    reference: MinimalSource, retrieved: list[MinimalSource]
) -> bool:
    """Whether any retrieved source covers a reference source."""
    return any(
        source.file_path == reference.file_path
        and _overlap_ratio(source, reference) >= IOU_THRESHOLD
        for source in retrieved
    )


def _answered_by_id(
    reference: RagDataset,
) -> dict[str, AnsweredQuestion]:
    """Index the reference questions that actually carry sources."""
    return {
        question.question_id: question
        for question in reference.rag_questions
        if isinstance(question, AnsweredQuestion) and question.sources
    }


def count_comparable(
    results: StudentSearchResults, reference: RagDataset
) -> int:
    """How many questions can be scored against the reference.

    Zero means the two files do not describe the same questions, which
    is worth reporting: an unscored run and a run that scored nothing
    both look like a recall of 0 otherwise.
    """
    answered = _answered_by_id(reference)
    return sum(
        1
        for entry in results.search_results
        if entry.question_id in answered
    )


def recall_at_k(
    results: StudentSearchResults, reference: RagDataset, k: int
) -> float:
    """Mean share of reference sources retrieved within the top k.

    Questions missing from either side are skipped rather than counted
    as failures, so a partial run still reports a meaningful figure.
    Use :func:`count_comparable` to tell an empty comparison apart from
    a genuine zero.
    """
    answered = _answered_by_id(reference)
    ratios: list[float] = []
    for entry in results.search_results:
        question = answered.get(entry.question_id)
        if question is None:
            continue
        top = entry.retrieved_sources[:k]
        found = sum(
            1 for source in question.sources if _is_found(source, top)
        )
        ratios.append(found / len(question.sources))
    return sum(ratios) / len(ratios) if ratios else 0.0
