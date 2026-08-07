"""Answer generation from retrieved context.

The retrieved sources are locations, not text, so the excerpts are read
back from disk with the ranges the retriever reported. They are then
placed in the model's context window and the model is asked to answer
from them alone.

Two choices are worth stating. Qwen3 emits a reasoning block by default;
at 0.6B parameters that block consumes the whole budget and the answer
never arrives, so thinking is disabled. And decoding is greedy, so the
same search results always produce the same answers.
"""

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_SOURCES = 10
DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_MAX_NEW_TOKENS = 256

NO_CONTEXT = "No source was retrieved for this question."

_SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. Use only the "
    "excerpts given to you. If they do not contain the answer, say so "
    "plainly. Never invent an API, a flag, a default value or a file "
    "name. Answer in at most three sentences."
)


class GenerationError(Exception):
    """The language model could not be loaded."""


def read_source_text(source: MinimalSource) -> str:
    """Read back the excerpt a source points at.

    An unreadable file yields an empty string rather than raising: one
    missing file must not stop a whole dataset run.
    """
    try:
        text = Path(source.file_path).read_text(
            encoding="utf-8", errors="ignore"
        )
    except OSError:
        return ""
    return text[
        source.first_character_index:source.last_character_index
    ]


class AnswerGenerator:
    """A local causal language model answering from retrieved context.

    Loading the weights is expensive, so one instance is meant to be
    reused across a whole dataset.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        """Load the tokenizer and the weights.

        Raises:
            GenerationError: If the model cannot be loaded, so that the
                CLI can report it instead of crashing.
        """
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32
            )
        except Exception as error:  # noqa: BLE001 - reported as-is
            raise GenerationError(
                f"cannot load {model_name}: {error}"
            ) from error
        self._model.eval()
        self.max_sources = max_sources
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens

    def _token_count(self, text: str) -> int:
        """Length of a string in model tokens."""
        return len(self._tokenizer(text).input_ids)

    def build_context(self, sources: list[MinimalSource]) -> str:
        """Assemble numbered excerpts that fit the token budget.

        Sources arrive best first, so filling greedily and stopping at
        the budget keeps the most relevant ones.
        """
        blocks: list[str] = []
        used = 0
        for source in sources[:self.max_sources]:
            excerpt = read_source_text(source).strip()
            if not excerpt:
                continue
            block = (
                f"[{len(blocks) + 1}] {source.file_path}\n{excerpt}"
            )
            cost = self._token_count(block)
            if blocks and used + cost > self.max_context_tokens:
                break
            blocks.append(block)
            used += cost
        return "\n\n".join(blocks)

    def answer(
        self, question: str, sources: list[MinimalSource]
    ) -> str:
        """Answer one question from its retrieved sources."""
        context = self.build_context(sources)
        if not context:
            return NO_CONTEXT
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Excerpts:\n\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer([prompt], return_tensors="pt")
        with torch.no_grad():
            # transformers types the loaded model as a base class that
            # does not line up with its own generate() signature.
            output = self._model.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs.input_ids.shape[1]:]
        return str(
            self._tokenizer.decode(generated, skip_special_tokens=True)
        ).strip()


def answer_search_results(
    generator: AnswerGenerator, results: StudentSearchResults
) -> StudentSearchResultsAndAnswer:
    """Answer every question of a search results file."""
    answered = [
        MinimalAnswer(
            question_id=entry.question_id,
            question=entry.question,
            retrieved_sources=entry.retrieved_sources,
            answer=generator.answer(
                entry.question, entry.retrieved_sources
            ),
        )
        for entry in tqdm(
            results.search_results, desc="Answering", unit="question"
        )
    ]
    return StudentSearchResultsAndAnswer(
        search_results=answered, k=results.k
    )


def save_answers(
    answers: StudentSearchResultsAndAnswer,
    save_directory: str | Path,
    filename: str,
) -> Path:
    """Write answers, creating the directory if needed."""
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        answers.model_dump_json(indent=2), encoding="utf-8"
    )
    return path
