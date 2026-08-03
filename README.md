_This project has been created as part of the 42 curriculum by hkanamit._

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about the
vLLM codebase.

A language model only knows what it was trained on, and retraining it to
add a private codebase is not realistic. This project takes the other
route: the corpus is indexed once, the snippets that actually answer a
question are retrieved at query time, and the answer is generated from
those snippets rather than from memory.

The pipeline has four stages:

1. **Indexing** — walk the corpus, cut every file into chunks, and
   persist the term statistics retrieval needs.
2. **Retrieving** — score every chunk against a question with BM25 and
   return the best source locations.
3. **Augmenting** — place the retrieved snippets in the model's context
   window.
4. **Generating** — produce a grounded answer from that context.

Retrieval quality is measured with recall@k against reference datasets.
The system currently reaches **87.0% recall@5 on docs questions**
(threshold 80%) and **58.6% on code questions** (threshold 50%).

> **Status:** stages 1–2 are implemented and measured. Answer generation
> (stages 3–4, the `answer` and `answer_dataset` commands) is not
> implemented yet.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) as the project and package manager
- The vLLM corpus placed at `data/raw/vllm-0.10.1/`

Model weights and the deep-learning stack can total several gigabytes,
so run the project from a location with enough free disk space.

### Installation

```bash
make install        # runs uv sync
```

### Running the pipeline

```bash
# 1. Index the corpus once (about 3 seconds)
uv run python -m src index --max_chunk_size 2000

# 2. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 3. Score the results locally
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Always scope `--save_directory` by dataset: the public datasets share
file names, so writing every run into the same folder would overwrite
previous results.

### Make targets

| Target        | Effect                                              |
| ------------- | --------------------------------------------------- |
| `install`     | `uv sync`                                            |
| `run`         | Run a command, e.g. `make run ARGS="search foo"`     |
| `debug`       | Same under `pdb`                                     |
| `clean`       | Remove `__pycache__` and cache directories           |
| `lint`        | `flake8 .` and `mypy .` with the mandatory flags     |
| `lint-strict` | `flake8 .` and `mypy . --strict`                     |

Both lint targets pass with no findings.

## System architecture

```
data/raw/vllm-0.10.1/
        │
        │  chunk.py      two chunking strategies, dispatched by extension
        ↓
   list[Chunk]           file path + character range + text
        │
        │  preprocess.py tokenize (shared with retrieval)
        │  indexer.py    context prefix, term counts, inverted index
        ↓
data/processed/index.pkl
        │
        │  retriever.py  BM25 scoring, batch search, self-evaluation
        ↓
data/output/search_results/<DatasetScope>/<dataset>.json
```

| Module          | Responsibility                                        |
| --------------- | ----------------------------------------------------- |
| `models.py`     | pydantic models exchanged between stages               |
| `load_json.py`  | Dataset loading, with all failures as `DatasetError`   |
| `chunk.py`      | Cut a file into chunks, keeping character offsets      |
| `preprocess.py` | Tokenization, shared by indexing and querying          |
| `indexer.py`    | Build, persist, load the index; BM25 search            |
| `retriever.py`  | Batch search, result I/O, recall@k                     |
| `cli.py`        | Python Fire commands, argument checking, reporting     |

The command layer holds no retrieval logic. Every command is a thin
wrapper over library functions, so the same pipeline could be driven by
an HTTP server or a notebook without duplicating code.

### Data models

`MinimalSource` (file path plus a character range) is the atom of the
system. `Chunk` extends it with the chunk text and converts back with
`to_source()`. Everything the grader reads — `StudentSearchResults` and
its entries — is built from those locations.

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so
two distinct strategies are implemented and dispatched by extension.

Every chunk records **the character range it covers in the original
file**. Retrieval is graded on the overlap between that range and the
reference one, so the offsets are the one thing that must never drift.

### Markdown and text — heading boundaries

Sections are cut at ATX headings. Oversized sections fall back to
paragraph boundaries, then to a hard split that guarantees the size
limit.

The strategy was chosen from the data rather than by intuition:
**79% of the reference docs spans begin with a Markdown heading**, and
53% end right before the next one. The reference answers were clearly
built section by section, so cutting on headings aligns our chunks with
theirs.

Two details matter:

- **Fenced code blocks are tracked.** A `# install the package` line
  inside a shell snippet matches the heading pattern exactly. Without
  fence tracking, 137 of 1,424 detected headings (10%) are false
  positives that split code blocks in half.
- **Blank sections are dropped.** Consecutive headings leave whitespace
  only ranges that can never match a query.

### Python — top-level definitions

`ast` is used to find the character range of every top-level function
and class; oversized classes are split again into their methods.

A regular expression is not enough here: it cannot tell a nested
function from a top-level one, cannot find where a definition ends when
the signature spans several lines, and matches `def` inside strings.

Two details matter:

- **Decorators are folded in.** `node.lineno` points at the `def` line,
  not at the decorators above it. The corpus holds 6,990 decorators
  that would otherwise fall outside their own chunk.
- **The gaps are kept.** Imports, module docstrings and module-level
  code live outside any definition. Indexing only the definitions loses
  **15% of the corpus text**.

Files that fail to parse fall back to fixed-size pieces instead of being
dropped. No file in vLLM currently fails, but indexing must not stop on
a single bad file.

### Why the pipeline has several stages

Each stage prevents a distinct failure, measured by removing it:

| Stage             | Removing it causes                                  |
| ----------------- | --------------------------------------------------- |
| Fence tracking    | 10% of detected headings are false                   |
| Paragraph split   | 8% of sections exceed 2000 chars (max 14,440)        |
| Greedy repacking  | 31% of chunks fall under 100 chars; 1.8× more chunks |
| Hard split        | 15 chunks still exceed the limit                     |
| Gap segments (py) | 15% of the corpus text is lost                       |
| Class splitting   | 13% of segments exceed 2000 chars                    |

Structure is honoured first and machine cuts are the last resort, so a
clean file yields clean section-sized chunks while a pathological one
still respects the limit. That limit is not negotiable: the grader
rejects any source longer than `max_context_length`, and a single
over-long source invalidates the whole output.

## Retrieval method

**BM25** over an inverted index.

The index stores raw statistics, not scores: a term maps to the list of
`(chunk index, term frequency)` pairs, alongside each chunk's token
count and location. The scoring formula is applied at query time, which
means TF-IDF could be swapped in without rebuilding anything.

For each query term, the score added to a chunk is

```
idf(term) × f × (k1 + 1) / (f + k1 × (1 − b + b × len / avg_len))
```

with `k1 = 1.5` and `b = 0.75`, and

```
idf(t) = log((N − df + 0.5) / (df + 0.5) + 1)
```

Two properties matter for this corpus. Term frequency **saturates**, so
a chunk that repeats an identifier twenty times does not outrank one
that explains it. And chunk length is **normalised**, which matters when
chunk sizes vary from a two-line gap to a full 2000-character section.

Chunks that contain no query term are never touched. A typical question
scores about half the index and skips the rest.

### Tokenization

One function, `preprocess.tokenize`, is used by both indexing and
querying. A term normalised one way at index time and another way at
query time can never match, and sharing the function is what makes that
impossible.

Tokens are `[a-z0-9_]+`, lowercased. **Underscores are kept inside
tokens on purpose**: code questions quote identifiers verbatim
(`load_lora_adapter`, `mm_kwargs`), and splitting on underscores would
destroy the strongest matching signal available.

### Context enrichment

Chunks are indexed with a short prefix that does not appear in their own
text: the parent directory and file name, plus the enclosing `class`
line when there is one.

This addresses a structural problem. Splitting a large class into its
methods puts the class name in a different chunk from the method body,
so a question like *"the default value for `mlp_bias` in
`ExaoneGatedMLP`"* needs two words that never co-occur in any chunk.

The prefix feeds the term statistics only. **The reported character
range stays exactly the one the chunker produced**, which is what the
grader compares.

## Performance analysis

Measured on the public datasets, with one index built over the whole
corpus.

### Final results

| Dataset | recall@1 | recall@3 | **recall@5** | recall@10 | Threshold |
| ------- | -------- | -------- | ------------ | --------- | --------- |
| docs    | 0.600    | 0.830    | **0.870**    | 0.920     | 0.80 ✓    |
| code    | 0.354    | 0.545    | **0.586**    | 0.626     | 0.50 ✓    |

| Constraint            | Limit | Measured  |
| --------------------- | ----- | --------- |
| Indexing time         | 300 s | **3.2 s** |
| 200 questions         | 90 s  | **2.9 s** |

The index holds 25,421 chunks (23,787 from Python, 1,634 from Markdown
and text) over 1,969 files, with a vocabulary of 55,221 terms and a
mean chunk length of 77 tokens. It occupies 10.2 MB on disk and loads in
0.2 seconds.

### Chunk size and overlap

Measured on a Markdown-only index, so the figures are not directly
comparable with the final ones above; they show the trend.

| Max size | no overlap | +100  | +200  | +400  |
| -------- | ---------- | ----- | ----- | ----- |
| 300      | 84.0%      | 84.0% | 83.0% | —     |
| 500      | 85.0%      | 87.0% | 89.0% | 81.0% |
| 800      | 89.0%      | 91.0% | 88.0% | 87.0% |
| 1200     | 88.0%      | 89.0% | 91.0% | 88.0% |
| 2000     | 89.0%      | 91.0% | 89.0% | 87.0% |

Smaller chunks are not better here. At 300 characters recall drops
clearly: a chunk that small rarely carries both the words of the
question and the words of the answer. Between 800 and 2000 the
difference is within two questions out of a hundred.

A useful way to read a chunking configuration is its **ceiling**: the
share of reference spans that overlap *some* chunk, ignoring ranking.

| Configuration    | Ceiling |
| ---------------- | ------- |
| Fixed 2000       | 96%     |
| Fixed 500        | 100%    |
| Heading-based    | **100%** |

The heading strategy reaches 100%, so every remaining loss is a ranking
problem, not a chunking one. That is why chunking work stopped there.

The 4% lost by fixed 2000 comes from the IoU rule rather than from
missing the region: an overlap counts only above 0.05, so a reference
span of 12 characters inside a 2000-character chunk scores 0.006 and
fails even though the chunk contains it.

### Overlap was measured and rejected

Overlap is a reasonable idea and it does help a fixed-size chunker,
where it lifts the ceiling from 96% to 100%. It does not help here.

| Overlap | docs recall@5 | Ceiling |
| ------- | ------------- | ------- |
| none    | 89.0%         | 100%    |
| 200     | 89.0%         | 100%    |
| 400     | 87.0%         | 100%    |
| 600     | 87.0%         | 99%     |

The ceiling is already 100%, so there is nothing left to reach. Worse,
the hard 2000-character limit means the overlap has to be taken out of
the packing budget, which fragments sections that would otherwise fit in
a single chunk. Overlap is therefore not used.

### One index, not two

Early measurements used a Markdown-only index for docs questions and a
Python-only index for code questions. That is optimistic: the real
pipeline builds **one** index and both datasets query it.

| Index                     | docs  | code  |
| ------------------------- | ----- | ----- |
| Markdown only             | 89.0% | —     |
| Python only               | —     | 45.5% |
| Combined, no enrichment   | 85.0% | 45.5% |
| Combined, enrichment      | 87.0% | 58.6% |

Combining costs docs four points, because Python chunks now compete for
the top five. Checking this before finishing was worth it: without
enrichment the combined index fails the code threshold.

### Context enrichment and path length

Enrichment is what carries code questions over the threshold, and the
shape of the path prefix turned out to matter as much as its presence.

| Path form in the prefix           | docs  | code  |
| --------------------------------- | ----- | ----- |
| Full path                         | 82.0% | 56.6% |
| Relative to the corpus root       | 83.0% | 57.6% |
| **Parent directory + file name**  | **87.0%** | **58.6%** |

`data raw vllm 0 10 1` is identical for every file, so it carries no
signal while still inflating chunk length — which BM25 penalises through
its length normalisation. Trimming the prefix improved *both* datasets.

### Where the remaining loss is

Ranking positions of the correct chunk, over the 99 code questions:

| Position | Questions |
| -------- | --------- |
| 1        | 32        |
| 2–5      | 13        |
| 6–10     | 10        |
| 11–50    | 12        |
| 51+      | 29        |
| Not found| 3         |

The 22 questions ranked between 6 and 50 are close, and most of them ask
for a default value of a named parameter in a named class. The far tail
is different in nature: those questions target one row of a large table,
where the query words appear dozens of times across the file and the
correct span is only a fragment of it.

## Design decisions

**Only `.py`, `.md` and `.txt` are indexed.** The subject leaves the
choice of files open. Every reference source is a `.md` (97), a `.txt`
(3) or a `.py` (99); no other extension is cited. The corpus also holds
403 JSON files, mostly machine-generated fixtures, which would inflate
the index and add noise for no measured gain.

**BM25 rather than TF-IDF.** Both were implemented over the same index.
BM25 was clearly better out of the box, mainly because term frequency
saturation and length normalisation suit chunks whose sizes differ by
two orders of magnitude.

**The index stores counts, not scores.** Keeping the statistics raw
means the scoring formula is a query-time concern. Swapping BM25 for
TF-IDF, or adding a second ranking to fuse with, needs no rebuild.

**Underscores are kept in tokens.** Splitting `load_lora_adapter` into
three common words would remove the only discriminating term in many
code questions.

**Enrichment changes what is indexed, never what is reported.** The
character range written to the output file is always the one the chunker
produced.

**Deterministic output.** Files are walked in sorted order and score
ties are broken by chunk index, so two runs on the same corpus produce
byte-identical results.

**Failures are typed and narrow.** `DatasetError` and `IndexLoadError`
give the CLI exactly two exceptions to catch, and every command exits
with a message instead of a traceback.

## Challenges faced

**Character offsets are the whole contract.** Grading compares file
paths verbatim and character ranges by overlap, so a chunker that loses
its offsets produces output that cannot be scored at all. `ast` reports
line numbers, not character positions, and `col_offset` is a UTF-8 byte
offset rather than a character index. The chunker therefore builds a
line-start table once per file and works in line units only. Three
invariants are checked over the corpus: the text of every chunk equals
`text[first:last]`, no chunk exceeds the limit, and every character of
every file belongs to some chunk.

**The reference spans for code do not follow the code structure.** Only
41% start at a `def`, `class` or decorator, and only 22% start at the
beginning of a line; some are 12 characters long and begin mid
docstring. Structural chunking cannot align with them. What saves the
metric is that overlap, not equality, is required — so semantically
coherent chunks are still the right target, they simply have to *cover*
the reference region.

**A field name typo that fails silently.** `first_charactor_index`
instead of `first_character_index` produces a JSON key the grader does
not recognise. Nothing raises; the score is simply zero. Serialising one
model and reading the output was enough to catch it, and it is the
reason the models are exercised early.

**Distinguishing "zero recall" from "nothing compared".** The evaluation
skips questions it cannot match by id, so comparing a docs result file
against the code reference reported a clean `0.000`. The command now
counts comparable questions first and refuses to report a figure when
none exist.

**`flake8 .` and `mypy .` scan everything.** The mandated lint commands
take no flags, and run as-is they report 36,176 findings from the
virtual environment and from vLLM itself. The exclusions live in
`.flake8` and in `pyproject.toml` so that the commands stay exactly as
the subject writes them.

## Example usage

Single query:

```console
$ uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter?" --k 3
1. data/raw/vllm-0.10.1/docs/features/lora.md [4695:6100]
2. data/raw/vllm-0.10.1/docs/features/lora.md [6100:8082]
3. data/raw/vllm-0.10.1/vllm/plugins/lora_resolvers/README.md [342:830]
```

The reference span for that question is `lora.md [4695:6098]`.

Indexing:

```console
$ uv run python -m src index --max_chunk_size 2000
Indexing: 100%|████████████████████| 1969/1969 [00:03<00:00, 623.54file/s]
Ingestion complete! 25421 chunks saved under data/processed/
```

Batch search and local scoring:

```console
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
Searching: 100%|██████████████████| 100/100 [00:00<00:00, 158.31question/s]
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Evaluated 100 questions
Recall@1: 0.600
Recall@3: 0.830
Recall@5: 0.870
Recall@10: 0.920
```

Degenerate input is reported, never raised:

```console
$ uv run python -m src search "" --k 5
No source found.

$ uv run python -m src search "lora" --processed_directory /nowhere
error: no index at /nowhere/index.pkl: run the index command first
```

## Resources

### On retrieval

- Manning, Raghavan and Schütze, *Introduction to Information
  Retrieval*, Cambridge University Press, 2009 — chapters 1 and 2 cover
  the inverted index, tokenization and term normalisation, which is the
  foundation this project is built on.
  <https://nlp.stanford.edu/IR-book/>
- Robertson and Zaragoza, *The Probabilistic Relevance Framework:
  BM25 and Beyond*, 2009 — the origin of the scoring function used here.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks*, 2020. <https://arxiv.org/abs/2005.11401>

### On the corpus

- vLLM. <https://github.com/vllm-project/vllm>
- Kwon et al., *Efficient Memory Management for Large Language Model
  Serving with PagedAttention*, 2023. <https://arxiv.org/abs/2309.06180>

### Tooling

- uv. <https://docs.astral.sh/uv/>
- Python Fire. <https://github.com/google/python-fire>
- pydantic. <https://docs.pydantic.dev/>
- Python `ast` module.
  <https://docs.python.org/3/library/ast.html>

### Use of AI

AI assistance (Claude) was used in the following ways:

- **Data analysis.** Scripts that measure the reference datasets — span
  length distributions, how often a reference span starts at a heading
  or a definition, the ranking position of the correct chunk. These
  measurements drove the chunking design instead of intuition.
- **Experiment harnesses.** The throwaway scripts behind the tables in
  *Performance analysis*: the chunk size and overlap grid, the
  separate-versus-combined index comparison, the path-prefix variants.
- **Implementation.** Drafting and reviewing the modules under `src/`,
  in particular the offset handling in `chunk.py` and the BM25 scoring
  in `indexer.py`.
- **Explanation.** Walking through unfamiliar parts of the standard
  library, mainly `ast` and the character-offset problem.

Everything AI produced was measured or read before being kept. Two of
its suggestions were rejected on the evidence: chunk overlap, which the
grid showed to be neutral or harmful here, and prefixing headings to
Markdown chunks, which changed nothing. Design choices that survived did
so because a number moved.
