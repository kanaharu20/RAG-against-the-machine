"""Chunking strategies for the corpus.

Two distinct strategies are implemented, as the subject requires: a
Python file and a Markdown page do not break apart the same way.

* Markdown / text is cut on heading boundaries, because a heading marks
  the start of a self-contained section.
* Python is cut on top-level definitions, using the ``ast`` module, so
  that a function or a class stays whole.

Both keep the character range each chunk covers in the original file:
retrieval is graded on the overlap between that range and the reference
one, so losing the offsets would make the output impossible to produce.
"""

import ast
import re
from pathlib import Path

from .models import Chunk

Span = tuple[int, int]

_HEADING = re.compile(r"^#{1,6} ")
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_FENCES = ("```", "~~~")
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

DEFAULT_MAX_CHUNK_SIZE = 2000


# --------------------------------------------------------------------
# Generic span helpers
# --------------------------------------------------------------------
def _line_starts(text: str) -> list[int]:
    """Return the character offset at which each line begins."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_spans(text: str, start: int, end: int) -> list[Span]:
    """Split ``text[start:end]`` into one span per line."""
    spans: list[Span] = []
    position = start
    while position < end:
        newline = text.find("\n", position, end)
        stop = end if newline == -1 else newline + 1
        spans.append((position, stop))
        position = stop
    return spans


def _paragraph_spans(text: str, start: int, end: int) -> list[Span]:
    """Split ``text[start:end]`` on blank lines."""
    spans: list[Span] = []
    cursor = start
    for match in _PARAGRAPH.finditer(text, start, end):
        spans.append((cursor, match.end()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _pack(units: list[Span], max_chunk_size: int) -> list[Span]:
    """Greedily merge consecutive units while they fit in the budget."""
    packed: list[Span] = []
    current: Span | None = None
    for start, end in units:
        if current is None:
            current = (start, end)
        elif end - current[0] <= max_chunk_size:
            current = (current[0], end)
        else:
            packed.append(current)
            current = (start, end)
    if current is not None:
        packed.append(current)
    return packed


def _hard_split(spans: list[Span], max_chunk_size: int) -> list[Span]:
    """Cut any span that is still too long into fixed-size pieces."""
    cut: list[Span] = []
    for start, end in spans:
        while end - start > max_chunk_size:
            cut.append((start, start + max_chunk_size))
            start += max_chunk_size
        if end > start:
            cut.append((start, end))
    return cut


def _to_chunks(
    text: str, file_path: str, spans: list[Span]
) -> list[Chunk]:
    """Materialise spans into chunks, dropping the blank ones."""
    chunks: list[Chunk] = []
    for start, end in spans:
        body = text[start:end]
        if body.strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=body,
                )
            )
    return chunks


# --------------------------------------------------------------------
# Strategy 1: Markdown / text
# --------------------------------------------------------------------
def _heading_offsets(text: str) -> list[int]:
    """Offsets of ATX headings that sit outside fenced code blocks.

    The fence state matters: a ``# comment`` inside a Python snippet
    looks exactly like a level-one heading.
    """
    offsets: list[int] = []
    position = 0
    inside_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(_FENCES):
            inside_fence = not inside_fence
        elif not inside_fence and _HEADING.match(line):
            offsets.append(position)
        position += len(line)
    return offsets


def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Cut a Markdown or text document on its heading boundaries."""
    marks = sorted(set([0, len(text)]) | set(_heading_offsets(text)))
    spans: list[Span] = []
    for start, end in zip(marks, marks[1:]):
        if end - start <= max_chunk_size:
            spans.append((start, end))
        else:
            units = _paragraph_spans(text, start, end)
            spans.extend(_pack(units, max_chunk_size))
    return _to_chunks(text, file_path, _hard_split(spans, max_chunk_size))


# --------------------------------------------------------------------
# Strategy 2: Python code
# --------------------------------------------------------------------
def _node_span(node: ast.stmt, starts: list[int], length: int) -> Span:
    """Character range of a node, decorators included.

    ``node.lineno`` points at the ``def`` line, not at the decorators
    above it, so they have to be folded in explicitly.
    """
    first_line = node.lineno
    for decorator in getattr(node, "decorator_list", []):
        first_line = min(first_line, decorator.lineno)
    last_line = getattr(node, "end_lineno", None) or node.lineno
    start = starts[first_line - 1]
    end = starts[last_line] if last_line < len(starts) else length
    return start, end


def _segments(
    body: list[ast.stmt],
    starts: list[int],
    start: int,
    end: int,
    length: int,
) -> list[tuple[int, int, ast.stmt | None]]:
    """Cover ``[start, end)`` with definitions and the gaps between.

    The gaps matter: imports, module docstrings and module-level code
    live outside any definition and would be lost otherwise.
    """
    segments: list[tuple[int, int, ast.stmt | None]] = []
    cursor = start
    for node in body:
        if not isinstance(node, _DEFINITIONS):
            continue
        node_start, node_end = _node_span(node, starts, length)
        if node_start < cursor:
            continue
        if node_start > cursor:
            segments.append((cursor, node_start, None))
        segments.append((node_start, node_end, node))
        cursor = node_end
    if cursor < end:
        segments.append((cursor, end, None))
    return segments


def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Cut a Python source file on its top-level definitions.

    A file that does not parse falls back to fixed-size pieces rather
    than being dropped: the corpus holds templates and snippets that
    are not valid Python.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return _to_chunks(
            text,
            file_path,
            _hard_split([(0, len(text))], max_chunk_size),
        )

    starts = _line_starts(text)
    length = len(text)
    pending = _segments(tree.body, starts, 0, length, length)
    spans: list[Span] = []
    while pending:
        start, end, node = pending.pop(0)
        if end - start <= max_chunk_size:
            spans.append((start, end))
        elif isinstance(node, ast.ClassDef):
            pending = _segments(
                node.body, starts, start, end, length
            ) + pending
        else:
            units = _line_spans(text, start, end)
            spans.extend(_pack(units, max_chunk_size))
    spans.sort()
    return _to_chunks(text, file_path, _hard_split(spans, max_chunk_size))


# --------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------
def chunk_file(
    path: str | Path,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Chunk one file, picking the strategy from its extension.

    Unreadable files yield no chunk instead of raising: indexing walks
    thousands of files and must not stop on one of them.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if file_path.suffix.lower() == ".py":
        return chunk_python(text, str(path), max_chunk_size)
    return chunk_markdown(text, str(path), max_chunk_size)
