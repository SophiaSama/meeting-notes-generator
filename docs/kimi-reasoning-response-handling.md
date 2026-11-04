# Kimi Reasoning-Model Response Handling

## Status

- **Date:** 2026-09-28
- **Scope:** `meeting_notes/llm.py`, `meeting_notes/pipeline.py`, and `tests/test_llm_service.py`
- **Decision:** Keep Chat Completions as the default API. Do not migrate to the Responses API until the configured provider is proven to support it.

## Summary

Calls to `Kimi-K2.6` could consume the entire completion-token budget on internal reasoning and return no visible answer. An attempted response-normalization change then introduced a separate regression: responses whose `completion_tokens_details` field was JSON `null` raised `AttributeError: 'NoneType' object has no attribute 'get'`, including responses from models that otherwise returned valid content.

The regression was fixed by treating optional nested usage fields as nullable. The application continues to separate reasoning from visible answer text and never publishes reasoning as meeting notes.

The original Kimi token-budget behavior is not fully solved by this parser fix. It requires provider-specific reasoning controls, a larger output budget, shorter inputs, or a bounded retry strategy.

## Observed Failures

### Kimi reasoning exhausted the completion budget

Representative error:

```text
RuntimeError: Model 'Kimi-K2.6' returned an empty response
(finish_reason='length' - the model hit the max_tokens limit before
producing visible content. Increase max_tokens or shorten the transcript.,
usage={'completion_tokens': 2200, 'prompt_tokens': 2629,
'total_tokens': 4829, 'completion_tokens_details': None, ...}).
```

Important properties of this response:

- `finish_reason` is `length`.
- The entire configured completion budget was consumed.
- The visible `message.content` is empty.
- A reasoning model may have generated internal reasoning before reaching the visible answer.
- Other models do not reproduce the issue because they do not use the output budget in the same way.

Increasing `max_tokens` may help, but it is not a deterministic fix: a reasoning model may use the additional budget for more reasoning.

### Null usage details caused a regression

The attempted normalizer used this pattern:

```python
usage.get("completion_tokens_details", {}).get("reasoning_tokens")
```

The default `{}` only applies when the key is absent. If the provider explicitly returns:

```json
{
  "completion_tokens_details": null
}
```

the first call returns `None`, and the second `.get()` raises `AttributeError`.

This parser exception occurred before valid content could be returned or an intentional empty-response error could be raised. It therefore affected models beyond Kimi.

## Implemented Fix

`LLMService._build_llm_response()` now normalizes the nested field before reading from it:

```python
completion_details = (
    usage.get("completion_tokens_details") if isinstance(usage, dict) else None
)
reasoning_tokens = (
    completion_details.get("reasoning_tokens")
    if isinstance(completion_details, dict)
    else None
)
```

The resulting behavior is:

1. A response with visible content succeeds even when `usage` or `completion_tokens_details` is `None`.
2. A response with no visible content still produces a controlled `RuntimeError` containing its finish reason and usage data.
3. Separate `reasoning_content` is retained for diagnostics but is not returned as the meeting-notes answer.
4. Structured content excludes parts identified as reasoning or thinking.

Prompt instructions were also strengthened to request only the final summary or notes. This can reduce unwanted visible commentary, but it should not be treated as a mechanism for disabling a model's internal reasoning.

## Test Coverage and Verification

Regression tests in `tests/test_llm_service.py` cover:

- visible content with `completion_tokens_details=None`;
- visible content with `usage=None`;
- Kimi reasoning-only output with `finish_reason="length"`;
- exclusion of reasoning/thinking structured content parts;
- existing synchronous and batch parsing behavior.

Verification performed after the fix:

```text
Focused LLM tests: 10 passed
Full test suite:    105 passed
Python compilation: passed
git diff --check:   passed
```

## API Decision: Chat Completions vs Responses

### Decision

Continue using Chat Completions for the current implementation.

### Rationale

- The project and its batch requests currently target `/v1/chat/completions`.
- The app targets any OpenAI-compatible provider. Chat Completions compatibility does not guarantee `/responses` compatibility.
- Provider extensions such as Kimi's `reasoning_content` may not map identically through the Responses API or an intermediate proxy.
- Existing batch behavior would also need separate verification and migration.
- The Responses API represents output and truncation differently, but it does not inherently prevent a reasoning model from exhausting its output-token budget before producing visible text.

### Conditions for reconsidering

Introduce a transport abstraction and consider the Responses API only after an authenticated capability test confirms that the configured provider supports:

1. `/responses` for `Kimi-K2.6` and all other configured models;
2. reasoning controls and their exact request schema;
3. text extraction from successful and incomplete responses;
4. equivalent batch processing, or an accepted alternative for batch workloads;
5. consistent token usage and error metadata.

Migration should be a separate change from response-parser bug fixes.

## Output-Budget Recovery (implemented 2026-09-29)

### Incident

With a single ×1.5 retry (4000 → 6000 tokens), Kimi-K2.6 still failed a
`segment_summary` call. On a 2752-token input it used all 6000 output tokens
thinking and wrote nothing visible, so the whole run aborted. Another segment
of the same size finished after 4252 tokens. How much Kimi thinks does not
track the input size, and a bigger budget alone doesn't guarantee a result.

### Design constraint

Kimi K3 is an always-thinking model: thinking **cannot** be turned off. So the
fix must not depend on reasoning control. It has to cope with a model that
may think for a long time.

### Recovery chain

Only `LLMOutputLimitError` (no visible content with `finish_reason="length"`)
triggers this chain. Auth, network and other errors are raised unchanged.

| Step | Stages | Action |
|------|--------|--------|
| 1 | all | Primary model with a model-aware stage budget |
| 2 | all | One retry on the primary model with `ceil(budget × LLM_NOTES_LENGTH_RETRY_TOKEN_MULTIPLIER)`, capped at `LLM_NOTES_LENGTH_RETRY_MAX_TOKENS` (32000) |
| 3 | map only | Split the segment in half (paragraph/sentence boundary) and summarise each half on the primary model |
| 4 | all | Send once to `LLM_NOTES_FALLBACK_MODEL` (default `gpt-4o`). In the map stage this applies per half |
| 5 | map only | Pass the raw segment text to the reduce step, marked `(raw transcript – summary unavailable)`. The reduce prompt tells the model to extract facts from it directly and mention it under Coverage Notes |

If a reduce or final-notes call is still unrecovered after step 4, the run
fails with the original `LLMOutputLimitError`. Every recovery step that was
taken is listed under **LLM output-limit recovery** in the notes' Artifacts
section.

### Supporting changes

- **Model-aware budgets:** when the stage env vars are unset, reasoning models
  (`LLM_REASONING_MODEL_PREFIXES`, default `kimi,o1,o3,o4,deepseek-r1,deepseek-reasoner`)
  get 16000 / 16000 / 24000 tokens for segment / reduce / final. Explicit
  `LLM_NOTES_*_MAX_TOKENS` values always take precedence.
- **Timeout:** reasoning models get a 300s HTTP timeout instead of 120s.
  Override it with `LLM_REQUEST_TIMEOUT_SECONDS`.
- **Reasoning effort (UI / MCP):** the "Reasoning effort" dropdown
  (`none|minimal|low|medium|high`, default from `LLM_REASONING_EFFORT`) sends
  `{"reasoning": {"effort": "<value>"}}` to every model. On HTTP 400/422 the
  request is silently resent without it and that (model, effort) pair is
  skipped for the rest of the process. `none` sends nothing and suppresses the
  env-based control below.
- **Optional reasoning control:** `LLM_REASONING_CONTROL_EXTRA_BODY` (JSON,
  empty by default) is used only when no reasoning effort is specified, and is
  sent only to reasoning models, e.g. `{"thinking": {"type": "disabled"}}` for
  Kimi K2.x. If the provider returns HTTP 400/422 for it, the request is resent
  once without it, and the parameter is disabled for that model for the rest
  of the process.
- **Diagnostics:** the call log now includes `reasoning_chars` (length only,
  never the content), `elapsed_s` and `reasoning_control`. Some providers report
  `reasoning_tokens=None`, so `reasoning_chars` may be the only indication of how
  much of the budget went to thinking.

### Remaining follow-up

- Confirm that `gpt-4o` (or your `LLM_NOTES_FALLBACK_MODEL`) is a model id your provider serves.
- Tune the reasoning-model budgets from real K2.6/K3 `reasoning_chars`
  and `output_tokens` logs.
- Once the provider's schema for Kimi reasoning control is confirmed, set
  `LLM_REASONING_CONTROL_EXTRA_BODY` for K2.x.

## Operational Troubleshooting

When this failure reappears:

1. Distinguish a parser failure from a model truncation failure.
   - `AttributeError ... NoneType ... get` indicates unsafe response parsing.
   - `RuntimeError ... finish_reason='length'` indicates a handled model-output limitation.
2. Confirm the exact model and requested `max_tokens` for the failing pipeline stage.
3. Check whether visible `message.content` is empty and whether `reasoning_content` is present.
4. Compare `completion_tokens` with the requested output limit.
5. Test provider-specific reasoning controls directly against the configured provider before changing application defaults.
6. Run:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests\test_llm_service.py -q
   .venv\Scripts\python.exe -m pytest -q
   ```

## Guardrails

- Never use reasoning content as a fallback answer.
- Never assume optional OpenAI-compatible response fields are non-null.
- Do not apply unbounded retries to reasoning models.
- Do not switch API transports based only on SDK support; verify provider and model support.
- Keep parsing, model-budget mitigation, and API migration as separate changes so failures remain diagnosable.