"""Command-line interface, built with Python Fire.

Every command is a thin wrapper over the library functions in the other
modules: the CLI validates its arguments, reports failures, and formats
output, but holds no retrieval logic of its own. That keeps the same
pipeline drivable by anything else later.

All input and output paths are arguments with defaults, never hard
coded values, because the evaluator points them at its own folders.
"""

import sys
from pathlib import Path
from typing import NoReturn

import fire

from .chunk import DEFAULT_MAX_CHUNK_SIZE
from .generator import (
    DEFAULT_MAX_SOURCES,
    AnswerGenerator,
    GenerationError,
    answer_search_results,
    save_answers,
)
from .indexer import (
    IndexLoadError,
    build_index,
    load_index,
    save_index,
)
from .load_json import DatasetError, load_dataset
from .retriever import (
    DEFAULT_K,
    count_comparable,
    load_search_results,
    recall_at_k,
    save_search_results,
)
from .retriever import search_dataset as run_search_dataset

RAW_DIRECTORY = "data/raw"
PROCESSED_DIRECTORY = "data/processed"
SEARCH_OUTPUT_DIRECTORY = "data/output/search_results"
ANSWER_OUTPUT_DIRECTORY = "data/output/search_results_and_answer"
REPORTED_K = (1, 3, 5, 10)


def _fail(message: str) -> NoReturn:
    """Report a failure and stop, instead of raising a traceback."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


class RagCli:
    """Retrieval-Augmented Generation over the vLLM corpus."""

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_directory: str = RAW_DIRECTORY,
        processed_directory: str = PROCESSED_DIRECTORY,
    ) -> str:
        """Ingest the corpus and persist the index.

        Args:
            max_chunk_size: Upper bound on the size of a chunk.
            raw_directory: Directory holding the corpus.
            processed_directory: Where the index is written.
        """
        if max_chunk_size <= 0:
            _fail("max_chunk_size must be a positive integer")
        if not Path(raw_directory).is_dir():
            _fail(f"no corpus directory at {raw_directory}")
        index = build_index(raw_directory, max_chunk_size)
        if not index.sources:
            _fail(f"no indexable file under {raw_directory}")
        path = save_index(index, processed_directory)
        return (
            f"Ingestion complete! {len(index.sources)} chunks "
            f"saved under {path.parent}/"
        )

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        processed_directory: str = PROCESSED_DIRECTORY,
    ) -> str:
        """Return the top-k sources for a single query.

        Args:
            query: Free text question.
            k: Number of sources to return.
            processed_directory: Where the index lives.
        """
        try:
            index = load_index(processed_directory)
        except IndexLoadError as error:
            _fail(str(error))
        sources = index.search(str(query), k)
        if not sources:
            return "No source found."
        return "\n".join(
            f"{rank}. {source.file_path} "
            f"[{source.first_character_index}:"
            f"{source.last_character_index}]"
            for rank, source in enumerate(sources, 1)
        )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = DEFAULT_K,
        save_directory: str = SEARCH_OUTPUT_DIRECTORY,
        processed_directory: str = PROCESSED_DIRECTORY,
    ) -> str:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: Dataset of questions to search.
            k: Number of sources to retrieve per question.
            save_directory: Where the results file is written. Scope it
                by dataset: the public datasets share file names.
            processed_directory: Where the index lives.
        """
        try:
            index = load_index(processed_directory)
            dataset = load_dataset(dataset_path)
        except (IndexLoadError, DatasetError) as error:
            _fail(str(error))
        if not dataset.rag_questions:
            _fail(f"no question in {dataset_path}")
        results = run_search_dataset(index, dataset, k)
        path = save_search_results(
            results, save_directory, Path(dataset_path).name
        )
        return f"Saved student_search_results to {path}"

    def answer(
        self,
        query: str,
        k: int = DEFAULT_MAX_SOURCES,
        processed_directory: str = PROCESSED_DIRECTORY,
    ) -> str:
        """Answer a single query from the retrieved context.

        Args:
            query: Free text question.
            k: Number of sources to retrieve and hand to the model.
            processed_directory: Where the index lives.
        """
        try:
            index = load_index(processed_directory)
        except IndexLoadError as error:
            _fail(str(error))
        sources = index.search(str(query), k)
        try:
            generator = AnswerGenerator()
        except GenerationError as error:
            _fail(str(error))
        return generator.answer(str(query), sources)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = ANSWER_OUTPUT_DIRECTORY,
    ) -> str:
        """Answer every question of a search results file.

        Args:
            student_search_results_path: Output of search_dataset.
            save_directory: Where the answers file is written. Scope it
                by dataset: the public datasets share file names.
        """
        try:
            results = load_search_results(student_search_results_path)
        except DatasetError as error:
            _fail(str(error))
        if not results.search_results:
            _fail(f"no question in {student_search_results_path}")
        print(f"Loaded {len(results.search_results)} questions")
        try:
            generator = AnswerGenerator()
        except GenerationError as error:
            _fail(str(error))
        answers = answer_search_results(generator, results)
        path = save_answers(
            answers,
            save_directory,
            Path(student_search_results_path).name,
        )
        return (
            f"Saved student_search_results_and_answer to {path}"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> str:
        """Report recall@k against a ground-truth dataset.

        This is for local iteration only. The official figure is the
        one the reference executable computes.

        Args:
            student_search_results_path: A search results file.
            dataset_path: The matching AnsweredQuestions dataset.
        """
        try:
            results = load_search_results(student_search_results_path)
            reference = load_dataset(dataset_path)
        except DatasetError as error:
            _fail(str(error))
        compared = count_comparable(results, reference)
        if compared == 0:
            _fail(
                "no question in common: check that the results and the "
                "AnsweredQuestions dataset describe the same questions"
            )
        lines = [f"Evaluated {compared} questions"]
        lines += [
            f"Recall@{k}: {recall_at_k(results, reference, k):.3f}"
            for k in REPORTED_K
        ]
        return "\n".join(lines)


def main() -> None:
    """Dispatch a command with Python Fire."""
    fire.Fire(RagCli)
