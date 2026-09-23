---
title: "RAG Facts Check"
subtitle: "Automated Claim-Level Verification for Enterprise RAG"
author: "European Environment Agency (EEA)"
date: "September 2026"
---

## Verification Pipeline

![](images/pipeline_graph.png)

::: notes
When the chatbot generates an answer from European environmental documents, RAG Facts Check audits every claim in real time:
1. Ingests the LLM answer alongside matched source documents.
2. Decomposes the answer into atomic claims.
3. Crossmatches each claim against referenced documents using an LLM judge.
4. Produces grounded verification verdicts, direct evidence citations, and hallucination flags.
:::

## Claim-Level Verification & Evidence

::::: columns
::: {.column width="50%"}
### Verification & Grounding

- **Claim Extraction:** <small>Decomposes answer into atomic claims</small>
- **Source Grounding:** <small>Verifies claims against source documents</small>
- **Verbatim Evidence:** <small>Locates exact quote & offsets in sources</small>
- **Hallucination Detection:** <small>Flags ungrounded claims</small>
- **Quality Score:** <small>Calibrated score from supported claims</small>
:::
::: {.column width="50%"}
![](images/screenshot_placeholder.png)
:::
:::::

::: notes
The fact-checking engine provides automated verification for enterprise RAG:
- Extracts atomic factual claims from generated chatbot answers.
- Checks each claim against referenced source documents using an LLM judge.
- Locates verbatim evidence spans and character offsets in the source text.
- Accurately detects hallucinations by flagging statements that lack document grounding (Not Enough Info).
- Produces a calibrated quality score reflecting overall answer trustworthiness.
:::
