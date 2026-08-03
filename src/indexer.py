"""Corpus indexing and BM25 retrieval.

Indexing walks the corpus, cuts every file into chunks, and stores the
term statistics BM25 needs. The index is persisted so that retrieval
only pays for loading it, never for rebuilding it.

Chunks are indexed with a short context prefix: the directory and file
name, plus the enclosing ``class`` line when there is one. Those words
place a chunk in the corpus but do not appear in its own text, and
questions lean on them heavily ("in ExaoneGatedMLP", "for the keye
model"). The prefix only feeds the term statistics; the character
range that gets reported stays exactly the one the chunker produced.
"""

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .chunk import DEFAULT_MAX_CHUNK_SIZE, chunk_file
from .models import Chunk, MinimalSource
from .preprocess import tokenize

CORPUS_EXTENSIONS = (".py", ".md", ".txt")
INDEX_FILENAME = "index.pkl"
BM25_K1 = 1.5
BM25_B = 0.75

_CLASS_LINE = re.compile(r"^\s*class\s+\w+.*$", re.MULTILINE)


class IndexLoadError(Exception):
    """The persisted index is missing or cannot be read."""


@dataclass
class BM25Index:
    """Term statistics over the chunked corpus.

    Attributes:
        sources: Location of every chunk, in index order.
        lengths: Token count of every chunk, in index order.
        postings: Term to ``(chunk index, term frequency)`` pairs.
        average_length: Mean chunk length, used to normalise scores.
        k1: Saturation of the term frequency.
        b: Strength of the length normalisation.
    """

    sources: list[MinimalSource] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(
        default_factory=dict
    )
    average_length: float = 0.0
    k1: float = BM25_K1
    b: float = BM25_B

    def _idf(self, document_frequency: int) -> float:
        """Weight of a term, high when few chunks contain it."""
        total = len(self.sources)
        return math.log(
            (total - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1
        )

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the k best-scoring chunk locations for a query.

        Args:
            query: Free text question.
            k: Number of results wanted.

        Returns:
            At most ``k`` locations, best first. An empty or unknown
            query, or a non-positive ``k``, yields an empty list.
        """
        if k <= 0 or not self.sources:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self._idf(len(posting))
            for index, frequency in posting:
                length = self.lengths[index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                scores[index] += (
                    idf * frequency * (self.k1 + 1) / denominator
                )
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        return [self.sources[i] for i in ranked[:k]]


def iter_corpus_files(raw_directory: str | Path) -> list[Path]:
    """List the corpus files worth indexing, in a stable order.

    Only the extensions the reference questions actually cite are kept.
    Sorting makes two runs produce the same index, which keeps results
    reproducible.
    """
    root = Path(raw_directory)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in CORPUS_EXTENSIONS
    )


def _context_prefix(
    location: str, class_lines: list[tuple[int, str]], chunk: Chunk
) -> str:
    """Build the context words to index alongside a chunk."""
    parts = [location]
    if not chunk.text.lstrip().startswith("class"):
        enclosing = [
            line
            for start, line in class_lines
            if start <= chunk.first_character_index
        ]
        if enclosing:
            parts.append(enclosing[-1])
    return "\n".join(parts) + "\n"


def _location_words(path: Path) -> str:
    """Directory and file name as searchable words.

    Only the last two path components are used. The rest of the path is
    identical for every file, so it carries no signal while still
    inflating chunk length, which BM25 penalises.
    """
    return f"{path.parent.name} {path.stem}".replace("_", " ")


def build_index(
    raw_directory: str | Path,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> BM25Index:
    """Chunk the corpus and accumulate BM25 statistics.

    Args:
        raw_directory: Directory holding the corpus, usually data/raw.
        max_chunk_size: Upper bound on the size of a chunk.

    Returns:
        The index, ready to be searched or saved.
    """
    paths = iter_corpus_files(raw_directory)
    sources: list[MinimalSource] = []
    lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path in tqdm(paths, desc="Indexing", unit="file"):
        chunks = chunk_file(path, max_chunk_size)
        if not chunks:
            continue
        location = _location_words(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        class_lines = [
            (match.start(), match.group().strip())
            for match in _CLASS_LINE.finditer(text)
        ]
        for chunk in chunks:
            prefix = _context_prefix(location, class_lines, chunk)
            counts = Counter(tokenize(prefix + chunk.text))
            position = len(sources)
            sources.append(chunk.to_source())
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((position, frequency))
    average = sum(lengths) / len(lengths) if lengths else 0.0
    return BM25Index(
        sources=sources,
        lengths=lengths,
        postings=dict(postings),
        average_length=average,
    )


def save_index(
    index: BM25Index, processed_directory: str | Path
) -> Path:
    """Persist an index, creating the directory if needed."""
    directory = Path(processed_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INDEX_FILENAME
    with path.open("wb") as stream:
        pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_index(processed_directory: str | Path) -> BM25Index:
    """Load a persisted index.

    Raises:
        IndexLoadError: If the index is missing or unreadable, so that
            the CLI can report it instead of crashing.
    """
    path = Path(processed_directory) / INDEX_FILENAME
    try:
        with path.open("rb") as stream:
            index = pickle.load(stream)
    except OSError as error:
        raise IndexLoadError(
            f"no index at {path}: run the index command first"
        ) from error
    except (pickle.UnpicklingError, AttributeError, EOFError) as error:
        raise IndexLoadError(f"corrupted index at {path}") from error
    if not isinstance(index, BM25Index):
        raise IndexLoadError(f"unexpected index content at {path}")
    return index
