"""Text preprocessing shared by indexing and retrieval.

Both sides must produce the same tokens: a term that is normalised one
way at indexing time and another way at query time can never match.
Keeping a single function here is what guarantees that.

Underscores are kept inside tokens on purpose. Questions about the code
quote identifiers verbatim (``load_lora_adapter``, ``mm_kwargs``), so
splitting on them would destroy the strongest matching signal there is.
"""

import re

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    Args:
        text: Raw text, from a chunk or from a query.

    Returns:
        The tokens, in order of appearance. Empty input yields an empty
        list rather than raising, so degenerate queries stay harmless.
    """
    return _TOKEN.findall(text.lower())
