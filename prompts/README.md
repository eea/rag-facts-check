# LLM Prompts

Prompt templates used by the RAG fact-checking pipeline. Each file is a plain text template loaded at runtime by `rag_facts_check/prompts.py`.

## Template Variables

Files may contain these placeholders, substituted at runtime:

| Variable | Description | Used in |
|---|---|---|
| `{system_prompt}` | System instruction for the phase | `*-prompt.txt` |
| `{text}` | The RAG answer text | `claim-extraction-prompt.txt` |
| `{claim}` | The claim to verify | `*-verification-*-prompt.txt` |
| `{documents}` | Formatted source documents | `*-verification-*-prompt.txt` |

## Files

### Claim Extraction

| File | Role |
|---|---|
| [`claim-extraction-system.txt`](claim-extraction-system.txt) | System instruction: how to extract factual claims |
| [`claim-extraction-prompt.txt`](claim-extraction-prompt.txt) | User prompt template with `{system_prompt}` and `{text}` |

### Claim Verification — Standard

| File | Role |
|---|---|
| [`claim-verification-system.txt`](claim-verification-system.txt) | System instruction: how to verify a claim |
| [`claim-verification-prompt.txt`](claim-verification-prompt.txt) | User prompt template with explicit output format instructions |

### Claim Verification — Evidence-First

| File | Role |
|---|---|
| [`claim-verification-evidence-first-system.txt`](claim-verification-evidence-first-system.txt) | System instruction: multi-step evidence-first verification |
| [`claim-verification-evidence-first-prompt.txt`](claim-verification-evidence-first-prompt.txt) | User prompt template (simpler — steps are in the system prompt) |

## Editing

Edit any `.txt` file directly. Changes take effect on next server restart (or next import in development). No rebuild needed.
