# exercise-04 — LLM Vocab and Delegation Map · SOLUTION

Paired exercise: `lessons/mod-105-deep-learning-fundamentals/exercises/exercise-04-llm-vocab-and-delegation-map.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise is deliberately short on code and long on organised writing. The chapter-10 objective is "working familiarity with transformer LLMs and where to delegate to specialist tracks" — a written artefact. The reference solution therefore does two things:

1. **Shows the shape of a passing deliverable** — file layout, section headers, table skeletons, cross-linking convention, and a worked example for each part so the learner can see the quality bar.
2. **Defines the rubric** — what counts as a passing glossary entry, how many leaves the delegation tree needs, which three fields on a model card must *always* be filled and which may legitimately be "not disclosed."

Everything in the deliverable lives under `exercise-04/` and resembles a small documentation site. `README.md` is the entry point. The substantive pieces are in `glossary.md`, `delegation_map.md`, `model_cards.md`, and `prompt_playground/`.

The grader's checklist in §4 is the acceptance criteria in a different shape — a learner submission is judged by it, not by prose-counting.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| Single documentation site with `README.md` as the entry point | One long Markdown file | Mirrors how a real internal reference ends up being organised; each part is linkable and can be refreshed independently |
| Glossary entries require a primary source | Allow Wikipedia / blog posts | The chapter's "cite the paper or vendor doc, not a Medium post" rule; downstream consumers trust the glossary only if it is anchored |
| Delegation map has **≥ 8 leaves** and **≥ 5 scenarios** | Fewer, broader buckets | A tree with 4 leaves collapses into "LLM yes/no"; the point of chapter 10's map is that there are at least four specialist tracks worth naming separately |
| Model cards comparison uses a **fixed table schema** | Free-form summaries | The table is the comparison; writing three narratives makes it hard to spot the difference between open and closed cards (which is one of the pedagogical points) |
| Prompt playground does **one** API call with three temperatures | A chat loop or RAG stub | Scope control: the exercise is marking the boundary of `mod-105`, not crossing into LLM Engineering |
| No agent frameworks, no vector DBs | Any of LangChain/LlamaIndex/Haystack | Those belong in the LLM Engineering track; including them here conflates "learn the vocabulary" with "ship a product" |

## 2. Worked implementation

### 2.1 Project layout

```
exercise-04/
├── README.md                   # entry point, one-page summary, reflection answers
├── glossary.md                 # every term from chapter 10, in your words, cited
├── delegation_map.md           # decision tree + 5 scenarios + boundaries
├── model_cards.md              # three completed card summaries + comparison
└── prompt_playground/
    ├── call.py                 # the tiny API script
    ├── NOTES.md                # temperature sweep, token counts, cost, reflection
    └── requirements.txt        # one dep: the SDK you chose
```

### 2.2 `README.md` — entry point skeleton

```markdown
---
last_refreshed: 2026-11-04
cohort: mod-105 exercise-04
---

# LLM Vocab and Delegation Map

A personal reference for transformer LLM vocabulary and the delegation decisions
that come with them. Chapter 10 is the companion.

## Contents
- [Glossary](./glossary.md) — every term from chapter 10, defined in my words with a primary source.
- [Delegation map](./delegation_map.md) — decision tree, worked scenarios, "when not to use an LLM", escalation criteria.
- [Model cards](./model_cards.md) — three cards read and compared, plus a one-page recommendation.
- [Prompt playground](./prompt_playground/) — one API call, three temperatures, grounded in NOTES.md.

## One-page delegation summary
(Flowchart or prose answer to "a text task lands on my desk — where does it go?"
Should fit on one screen. Full tree lives in delegation_map.md.)

## Reflection

(Answers to the four reflection questions.)
```

### 2.3 `glossary.md` — worked entries and the quality bar

The glossary must cover every term in chapter 10's vocabulary list (reproduced in the exercise). Each entry has four required elements:

1. **One-sentence plain-language definition.**
2. **Two-to-four sentence expansion** covering the mechanics or the operational trade-off.
3. **At least one primary source** — paper or vendor doc, not a blog post.
4. **Cross-links** to related entries.

Two worked entries, to anchor the quality bar:

```markdown
### LoRA (Low-Rank Adaptation)

Train a small pair of low-rank matrices that get added to a frozen pretrained
weight matrix, instead of updating the full weights.

For each adapted `W` of shape `(d_out, d_in)`, LoRA inserts `ΔW = B @ A` where
`A: (r, d_in)` and `B: (d_out, r)`, with `r ≪ min(d_out, d_in)` — typically 4–64.
Only `A` and `B` are trained. At inference you can either keep the adapters
separate (swap behaviours per request by swapping checkpoints) or merge them
back into `W` for zero added latency. Trades a modest accuracy hit on the hard
tail of tasks for a ~10,000× reduction in trainable parameters and a correspondingly
cheaper fine-tuning run.

**Source:** Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language
Models*. <https://arxiv.org/abs/2106.09685>

**Related:** [[PEFT]], [[QLoRA]], [[fine-tuning]], [[parameters]].
```

```markdown
### Top-p (nucleus) sampling

Sample from the smallest subset of tokens whose cumulative probability exceeds
`p`, renormalised.

Introduced to fix a specific failure of top-k sampling: when the distribution
is peaky (one or two tokens dominate), a large `k` dilutes the sample with
implausible tokens; when the distribution is flat, a small `k` truncates
reasonable candidates. Top-p sizes the sampling pool to the local shape of the
distribution instead — "take enough tokens to cover 90% of the probability mass,
however many that is". Common defaults are `p=0.9` for creative tasks, `p=1.0`
(disabled) for deterministic / tool-calling paths. Pairs with a non-zero
temperature — at `temperature=0` the argmax is taken and top-p is moot.

**Source:** Holtzman, A. et al. (2020). *The Curious Case of Neural Text
Degeneration*. ICLR. <https://arxiv.org/abs/1904.09751>

**Related:** [[temperature]], [[top-k sampling]], [[logits]], [[beam search]].
```

**Full term list (from the exercise, verbatim)** — write one entry per line below:

- Token; tokenizer (BPE, WordPiece, SentencePiece); vocabulary.
- Context length / context window.
- Parameters; memory footprint at inference.
- Pre-training; base model; next-token prediction; masked language modelling.
- Post-training; supervised fine-tuning (SFT); preference optimisation (RLHF, DPO).
- Fine-tuning; parameter-efficient fine-tuning (PEFT); LoRA; QLoRA.
- Prompt; system prompt; few-shot examples; in-context learning.
- Sampling / decoding: temperature; top-k; top-p / nucleus; beam search.
- Logits; logprobs.
- Instruction-tuned / chat model vs. base model.
- Prompt engineering.
- RAG (retrieval-augmented generation); embedding model; vector database.
- Function calling / tool use; agent.
- KV cache; paged attention; prefix caching.
- Quantization (INT8, INT4, GPTQ, AWQ, bitsandbytes).
- FLOPs; training compute; inference compute.
- MoE (Mixture of Experts).
- Prompt tokens vs. completion tokens; billing.
- Distillation.
- Evaluation benchmarks (MMLU, GSM8K, HumanEval, HellaSwag, MT-Bench, chatbot arena).
- Hallucination.

**Suggested primary sources per term** — use these first; drop the blog post:

| Term | Primary source |
|---|---|
| BPE | Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*. <https://arxiv.org/abs/1508.07909> |
| WordPiece | Wu, Y. et al. (2016). *Google's Neural Machine Translation System*. <https://arxiv.org/abs/1609.08144> |
| SentencePiece | Kudo, T. & Richardson, J. (2018). *SentencePiece*. <https://arxiv.org/abs/1808.06226> |
| Next-token pre-training | Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training*. (GPT-1 report.) |
| Masked language modelling | Devlin, J. et al. (2019). *BERT*. <https://arxiv.org/abs/1810.04805> |
| RLHF | Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). <https://arxiv.org/abs/2203.02155> |
| DPO | Rafailov, R. et al. (2023). *Direct Preference Optimization*. <https://arxiv.org/abs/2305.18290> |
| LoRA | Hu, E. et al. (2021). <https://arxiv.org/abs/2106.09685> |
| QLoRA | Dettmers, T. et al. (2023). <https://arxiv.org/abs/2305.14314> |
| In-context learning (few-shot) | Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). <https://arxiv.org/abs/2005.14165> |
| Nucleus sampling | Holtzman, A. et al. (2020). <https://arxiv.org/abs/1904.09751> |
| Beam search | A standard NMT reference, e.g. Sutskever, I. et al. (2014). *Sequence to Sequence Learning with Neural Networks*. <https://arxiv.org/abs/1409.3215> |
| RAG | Lewis, P. et al. (2020). *Retrieval-Augmented Generation*. <https://arxiv.org/abs/2005.11401> |
| Paged attention (vLLM) | Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. <https://arxiv.org/abs/2309.06180> |
| GPTQ (quantization) | Frantar, E. et al. (2023). *GPTQ*. <https://arxiv.org/abs/2210.17323> |
| AWQ | Lin, J. et al. (2023). *AWQ*. <https://arxiv.org/abs/2306.00978> |
| bitsandbytes (INT8 LLM.int8()) | Dettmers, T. et al. (2022). *LLM.int8()*. <https://arxiv.org/abs/2208.07339> |
| Mixture of Experts | Shazeer, N. et al. (2017). *Outrageously Large Neural Networks*. <https://arxiv.org/abs/1701.06538> |
| Distillation | Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network*. <https://arxiv.org/abs/1503.02531> |
| MMLU | Hendrycks, D. et al. (2021). *Measuring Massive Multitask Language Understanding*. <https://arxiv.org/abs/2009.03300> |
| GSM8K | Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*. <https://arxiv.org/abs/2110.14168> |
| HumanEval | Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code*. <https://arxiv.org/abs/2107.03374> |
| HellaSwag | Zellers, R. et al. (2019). *HellaSwag*. <https://arxiv.org/abs/1905.07830> |
| MT-Bench / Chatbot Arena | Zheng, L. et al. (2023). *Judging LLM-as-a-Judge*. <https://arxiv.org/abs/2306.05685> |
| Hallucination (survey) | Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation*. <https://arxiv.org/abs/2202.03629> |

**Style constraint.** No LLM-generated glossary passes. If a paragraph reads
vague, symmetrical, and hedged (a dead giveaway: "LoRA is a popular technique
for fine-tuning"), rewrite it. The value of the glossary is that it is
opinionated where it should be — "default to AWQ over GPTQ for a Transformer
on a Hopper GPU; the reverse on older hardware" is a glossary line, "LoRA is a
powerful technique" is noise.

### 2.4 `delegation_map.md` — worked decision tree and scenarios

The decision tree answers a single question: **"a task involving text or
language just landed on my desk. Where does it go?"** It needs at least eight
leaves covering distinct outcomes.

```markdown
# Delegation Map

## Decision tree

Start: a task involving text or language just landed on my desk.

1. Does the task require generating free-form language output?
   - Yes → go to step 2.
   - No  → go to step 5.

2. Is the task **a solved classical NLP problem** (classification, NER,
   sentiment, intent)?
   - Yes → use a smaller model (BERT-class, fastText, or a tuned logistic
     regression on TF-IDF). This stays in `mod-105` / `mod-107`.
     [leaf A: STAY — small classifier]
   - No  → continue.

3. Does the task need **current / proprietary knowledge** the base model
   cannot have seen?
   - Yes → RAG. [leaf B: LLM Engineering — RAG]
   - No  → go to step 4.

4. Does the task have **high stakes** (regulatory, legal, medical, financial)?
   - Yes → delegate to LLM Engineering for human-in-the-loop design.
     [leaf C: LLM Engineering — guardrailed]
   - No  → delegate to LLM Engineering for a thin generation wrapper.
     [leaf D: LLM Engineering — generation]

5. Does the task need to **act on the world** (tool use, DB write, API call,
   agent)?
   - Yes → delegate to LLM Engineering for tool-use / agent design. Pair with
     MLOps for deployment. [leaf E: LLM Engineering + MLOps — agent]
   - No  → go to step 6.

6. Is the task about **training or fine-tuning a model from scratch** (not
   just a one-off SFT run)?
   - Yes → delegate to ML Infra / Distributed Training.
     [leaf F: ML Infra — training]
   - No  → go to step 7.

7. Is the task **not textual at all** (vision, speech, time series)?
   - Yes, vision → Applied Vision.
     [leaf G: Applied Vision]
   - Yes, speech → Applied Speech.
     [leaf H: Applied Speech]
   - Time series → stay in `mod-104` / `mod-107`.
     [leaf I: STAY — classical ML]

8. Is the task about **cost / serving / latency** for an already-trained model?
   - Yes → MLOps / Deployment.
     [leaf J: MLOps — deployment]

9. **Not an LLM problem at all.** If a `grep`, a regex, a SQL query, or a
   human in the loop solves the request with lower latency and no vendor
   dependency, do that.
   [leaf K: NOT AN LLM PROBLEM]
```

A well-formed tree has at least leaves A, B, C/D, E, F, G or H, I or J, K — that
is the "≥ 8 distinct outcomes" acceptance bar.

**Five worked scenarios** (expand each to 3–5 sentences in the deliverable):

| # | Scenario | Target leaf | One-line reasoning |
|---|---|---|---|
| 1 | "Add a chatbot to our customer support portal." | D or C (LLM Engineering) | Generation + likely retrieval over tickets + likely tool-use for refunds; a wrapper around a hosted model is faster to prod than fine-tuning. |
| 2 | "Our internal wiki is 40k pages; nobody can find anything." | B (LLM Engineering — RAG) | Classical IR + an LLM generator over retrieved chunks; a fine-tune is not required and would get stale on the first wiki edit. |
| 3 | "Mobile app Whisper is too slow." | J (MLOps — deployment) | The model is already chosen; the problem is quantisation, batching, and on-device inference. LLM Engineering is adjacent but secondary. |
| 4 | "Marketing wants an image generator on product pages." | Applied Vision track, not LLM track | Different modality; LLMs have multimodal variants but a dedicated image model is almost always the right default for a product surface. |
| 5 | "Flagged for hallucinated legal citations in summarisation." | C (LLM Engineering — guardrailed) | High-stakes output that must be verifiable; the fix is retrieval + citation + evaluator-in-the-loop, not a larger model. |

**"When not to use an LLM" section** — at least three items, grounded in the
chapter-10 decision map:

- The task is **solvable with a regex, a rule, or a SQL join** with lower
  latency and no vendor dependency.
- The task requires **a correctness guarantee**. LLMs do not emit proofs of
  correctness; if a wrong answer is unacceptable, the LLM must be downstream
  of a verifier.
- The task has **latency budgets below the model's time-to-first-token**.
  Interactive editor completions below 50ms rarely fit even a quantised
  7B model's first-token latency; use a smaller specialist model or a cache.

**"Escalation criteria" section** — at least four items. Signals that mean
"this has grown beyond `mod-105` — hand it off":

- **Cost per query above a threshold.** Pick a concrete number; a six-figure
  monthly API bill is a specialist problem (quantisation, batching, caching,
  self-hosting trade-off), not a prompt-engineering problem.
- **Latency floor the current stack can't meet.** First-token latency above
  the UX budget; p99 above the SLO. Delegates to MLOps / Deployment.
- **Data sensitivity that rules out third-party APIs.** PHI, PII, regulated
  trade secrets. Delegates to ML Infra for on-prem serving or a VPC-contained
  deployment.
- **Regulatory audit requirements.** HIPAA, GDPR DSARs, SOC 2 Type II. Needs
  logging, data-retention, and often model-provenance capabilities that sit
  in MLOps / Governance.
- **Capability outgrows the stack** — e.g., long context (>100k tokens) at
  high concurrency; tool-use at high fan-out; fine-tune required for domain
  jargon. Delegates to LLM Engineering (capability design) + ML Infra
  (training infrastructure).

### 2.5 `model_cards.md` — three cards, comparison table

The exercise requires **three cards**, from **at least two different provider
families**, with **a mix of open-weights and closed-API** if possible. The
comparison table is prescribed in the exercise; every row must be filled or
explicitly marked `not disclosed`.

**Template:**

```markdown
## <Model name and version>

| Field | Value |
|---|---|
| Model name and version | |
| Provider | |
| Release date | |
| Total parameters (and active, for MoE) | |
| Layers, heads, `d_model` | |
| Context length | |
| Tokenizer vocab size | |
| Positional encoding (learned / sinusoidal / RoPE / ALiBi) | |
| Attention variant (MHA / GQA / MQA) | |
| Training corpus (size in tokens; mixture, if disclosed) | |
| Post-training method (SFT + RLHF / DPO / other) | |
| Reported benchmark headlines | |
| License | |
| Known limitations (from the card itself) | |

**Source:** <link to the model card>

**Short paragraph on what the card does not disclose, and why that matters:**
...
```

**Suggested card pool** (pick three — one open-weight MoE, one open-weight
dense, one closed-API gives the richest comparison):

- Meta **Llama** family cards (Llama 3.x or newer). Open-weights, dense.
- **Mistral** — Mistral 7B, Mixtral-8x7B (MoE), or the newer Mistral Small /
  Medium / Large.
- Alibaba **Qwen** (Qwen2.5 or Qwen3).
- **DeepSeek** — DeepSeek-V3 or DeepSeek-R1 (reasoning model; good contrast).
- Google **Gemma 2** or **Gemma 3**. Open-weights.
- Anthropic **Claude** system card (closed-API).
- **OpenAI GPT-OSS** cards (open-weights) or GPT-4/5 cards (closed-API).

<!-- author note: specific parameter counts, architectural widths, context
lengths, and training-corpus sizes change with every model release; the
learner fills these in from the model card they are reading, which must be
dated. The solution does not pre-populate these numbers because they would
be wrong within a release cycle. -->

**One-page comparison paragraph** at the end of the file: given these three
cards, pick one for each of three tasks and justify:

- **Low-cost high-throughput classification** — typically the smallest
  open-weight model in the pool (Llama 3.x 8B-class or smaller), self-hosted;
  classification is a solved problem and the LLM cost model favours cheap
  tokens at scale.
- **Long document summarisation (100k+ token context)** — pick the card with
  the largest *useful* context window (not just the largest declared one;
  cards often declare long contexts that degrade past a fraction of that).
  Verify with a card's own long-context benchmark results if given.
- **Code-assistant integration** — pick the card with the strongest HumanEval
  / LiveCodeBench headline, optionally a code-specialised variant. Prefer a
  model with tool-calling support and a licence that permits commercial code
  generation.

Three fields that must always be filled, even on closed-API cards:

1. **Release date.**
2. **Context length.** (Quoted in the API docs if not the card itself.)
3. **License / commercial-use clause.** Even "proprietary, under provider
   terms" is a filled answer.

Three fields that are *usually* "not disclosed" on closed-API cards:

- Total parameters and layer/head widths.
- Training corpus composition and token count.
- Post-training methodology beyond "RLHF / safety training."

The *contrast* between an open-weights card ("here is the recipe") and a
closed-API card ("here is what it can do") is itself one of the pedagogical
points; the short paragraph under each table is where you call that out.

### 2.6 `prompt_playground/` — one API call, three temperatures

The playground is one script. Required elements: system prompt, 1–2 few-shot
examples, a user turn, three temperatures, token counts read back.

```python
# prompt_playground/call.py
import os
import sys
import anthropic  # or openai / ollama / google-generativeai — pick one

SYSTEM = (
    "You are a careful technical reviewer. When asked for a definition, "
    "respond with exactly two sentences: a plain-English one, and a mechanics one."
)

FEW_SHOT = [
    ("What is a KV cache in an LLM?",
     "At inference, the attention keys and values from previous tokens are "
     "saved so each new token only computes attention against the new query. "
     "That turns per-token decoding from O(L^2) to O(L), and makes long-context "
     "generation feasible on typical hardware."),
    ("What is temperature in sampling?",
     "Temperature divides the logits before the softmax, flattening (T>1) or "
     "sharpening (T<1) the token distribution. T=0 is argmax; T=1 is unmodified; "
     "T>1 increases variety at the cost of coherence."),
]

USER = "What is top-p (nucleus) sampling?"


def call(client: anthropic.Anthropic, model: str, temperature: float) -> dict:
    messages = []
    for q, a in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": USER})

    resp = client.messages.create(
        model=model, system=SYSTEM, messages=messages,
        temperature=temperature, max_tokens=256,
    )
    return {
        "text": "".join(b.text for b in resp.content if b.type == "text"),
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set; using Ollama fallback is recommended.",
              file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    for t in (0.0, 0.7, 1.2):
        out = call(client, model, temperature=t)
        print(f"\n=== temperature={t} ===")
        print(out["text"])
        print(f"[tokens: prompt={out['prompt_tokens']} completion={out['completion_tokens']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Checklist for the script itself:

- API key via environment variable only; **never** committed.
- System prompt + few-shot + user turn in that order.
- Reads back both `text` and the token-count fields from `resp.usage`.
- Runs the same prompt at three temperatures and prints side-by-side.

**`NOTES.md` structure** — must answer all four required questions:

```markdown
# Playground notes

## 1. Which fields from a Part C model card did I actually see?
(e.g., "Context length was never close to the limit; tokenizer vocab affected
the prompt-token count — my 120-word prompt tokenised to 180 tokens.")

## 2. What did the temperature sweep show?
- T=0.0: deterministic, identical across runs, close to the few-shot style.
- T=0.7: minor rewording, same information, best readability.
- T=1.2: structural drift — occasionally added a third sentence; one run
  substituted "beam search" terminology inappropriately.

## 3. Token counts and cost
- Prompt tokens per call: ~400.
- Completion tokens per call: ~90.
- Pricing (from the provider's current public page): $<X> per 1M prompt, $<Y>
  per 1M completion.
- Estimated cost for 10,000 similar calls: $<fill>.

## 4. Would I use this model for the task in production?
(Honest answer. "No, a fine-tuned 1B specialist would be cheaper" is a valid
conclusion; the point is to think in cost-per-call terms early.)
```

**Cost cap.** Budget $1 total. If test spend exceeds that, the prompt or the
temperature range is misconfigured.

**No-API fallback.** Use Ollama locally with a 1–7B open-weights model. Note
in `NOTES.md` that costs and rate limits were not exercised; the vocabulary
grounding is the goal.

## 3. Validation steps

1. **File layout check** — all five files exist (`README.md`, `glossary.md`,
   `delegation_map.md`, `model_cards.md`, `prompt_playground/call.py`).
2. **Glossary completeness** — every term from the chapter-10 list has a
   section; `grep -c "^### " glossary.md` is at least the term count.
3. **Glossary citation check** — every entry has at least one URL to a paper
   or vendor doc, not a blog post. One-line review: `grep "^\*\*Source:" glossary.md | wc -l`
   equals the entry count.
4. **Cross-link resolution** — `[[name]]` references resolve to a section in
   `glossary.md`. A quick `awk` or a Markdown linter catches dangling links.
5. **Delegation tree leaf count** — `grep "\[leaf " delegation_map.md | sort -u | wc -l`
   is at least 8.
6. **Model card schema completeness** — three tables with every row filled or
   explicitly marked `not disclosed`.
7. **Playground runs end to end** — `python prompt_playground/call.py` prints
   three temperature sections and three token-count footers.
8. **No secrets committed** — `git diff --cached | grep -iE "sk-|api[_-]key|bearer"`
   returns nothing. CI-friendly; reproducible.
9. **Reading check** — the `README.md` one-page summary is readable top-to-
   bottom in under three minutes. If not, it is not a one-page summary.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Glossary coverage | 3 | Every term in the chapter-10 list has an entry; no silent omissions |
| 2 | Glossary quality | 3 | Each entry has plain-language + expansion + primary source + cross-links; sources are papers or vendor docs, not blog posts |
| 3 | Glossary voice | 1 | At least one entry per section expresses an opinion the author stands behind; no LLM blandness |
| 4 | Delegation tree | 3 | ≥ 8 distinct leaves covering the specialist tracks named in chapter 10; the "not an LLM problem" leaf is present |
| 5 | Delegation scenarios | 2 | 5 scenarios with target leaf + one-line reasoning each |
| 6 | "When not to use an LLM" | 1 | ≥ 3 concrete items, each tied to a measurable signal |
| 7 | Escalation criteria | 2 | ≥ 4 items, each a measurable threshold |
| 8 | Model cards: three completed | 2 | Table schema used; "not disclosed" is an explicit value where applicable |
| 9 | Model cards: open vs. closed contrast | 1 | One short paragraph per card calling out what the card does not disclose and why |
| 10 | Model cards: one-page recommendation | 1 | Three tasks (cheap classification; long-doc summary; code assistant) each mapped to a card with reasoning |
| 11 | Playground script | 1 | Runs end to end; system + few-shot + user; three temperatures; token counts printed; API key from env |
| 12 | Playground NOTES | 1 | All four required questions answered; cost multiplied through to 10,000 calls |

A submission missing item 1 (glossary coverage) cannot score above 14 — the
glossary is the foundation; the rest of the exercise assumes it.

## 5. Common mistakes

- **Glossary entries lifted from an LLM.** Reads symmetrically ("X is a popular
  technique used by…"), no opinion, no primary source. The style constraint in
  the exercise is explicit; failing it is a rewrite, not a deduction.
- **Wikipedia or a blog post as the primary source.** Wikipedia is a useful
  secondary cite; it is not where a technical glossary gets anchored. The
  source table in §2.3 above is the first-pass list of primary sources.
- **A delegation tree with four leaves.** Collapses to "LLM yes/no". The tree
  with eight leaves forces you to name the specialist tracks, which is the
  pedagogical point.
- **Scenarios that all route to LLM Engineering.** If every scenario lands on
  the same leaf, the tree is not doing work. At least two scenarios should
  route elsewhere (e.g., MLOps for serving, Applied Vision for cross-modal).
- **Model cards from one provider only.** The contrast between open-weights
  and closed-API cards is required; three Llama variants do not satisfy it.
- **Numbers invented for the cards.** If the card doesn't say the parameter
  count, write "not disclosed" — do not guess. A card summary with fabricated
  numbers is worse than one with "not disclosed" rows.
- **Playground that does more than one API call.** A chat loop, a RAG stub, or
  an agent is out of scope. The exercise is marking the boundary; crossing it
  is a different track.
- **Playground with the API key hard-coded.** Exercise-blocker; the final
  `grep` check is prescribed in the acceptance criteria.
- **Playground cost not computed.** Multiplying token count × price × 10,000
  calls is the entire point of the cost question.
- **`README.md` that is longer than one page.** Defeats the stated purpose.
  If the one-page summary spans three screens, it is the long document; link
  to it from a real one-page summary.

## 6. References

**Standards and project-level documentation:**

- NIST AI Risk Management Framework 1.0 — `GOVERN 1.1` (roles and
  responsibilities are documented; relevant to the delegation map's
  "escalation criteria"), `MAP 2.3` (contextual risks are documented;
  relevant to "when not to use an LLM"). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — particularly `ML06:2023 AI Supply
  Chain Attacks` as the framing for "treat a pretrained checkpoint as a
  supply-chain input." <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` and `AML.T0051 LLM
  Prompt Injection` give concrete adversarial framings for several
  delegation-map decisions. <https://atlas.mitre.org/>

**Vendor documentation referenced throughout:**

- Anthropic model documentation. <https://docs.anthropic.com/>
- OpenAI model documentation. <https://platform.openai.com/docs/models>
- Hugging Face model card documentation and reference. <https://huggingface.co/docs/hub/model-cards>
- Meta Llama model cards (via Hugging Face or the Llama project). <https://ai.meta.com/llama/>
- Ollama model library. <https://ollama.com/library>

**Research references** (primary sources for glossary entries) — see the
suggested-source table in §2.3.

<!-- author note: specific model parameter counts, context lengths, and
release dates for the model cards section will go stale within a release
cycle; the learner fills them in from the current card at the time of
writing, and marks the `last_refreshed` date in the README front matter so
future reviewers can judge whether the entries need refreshing. -->
