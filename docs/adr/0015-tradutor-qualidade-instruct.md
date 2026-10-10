# ADR-0015 — Opção de qualidade do tradutor: Qwen3 30B-A3B Instruct

- **Status:** Aceito (complementa a [ADR-0005](0005-traducao-por-llm-local.md))
- **Data:** 2026-10-09

## Contexto
A ADR-0005 indicou `qwen3:30b-a3b` "com raciocínio desativado" como opção de qualidade. A tag instalada é a variante **Thinking-2507** (o `ollama show` lista a capability `thinking`), e nessa variante o raciocínio não desliga: cada fala gastaria tokens raciocinando e a saída JSON ficaria menos previsível.

## Decisão
A opção de qualidade passa a ser **`qwen3:30b-a3b-instruct-2507-q4_K_M`** (sem modo de raciocínio), baixada via Ollama para o SSD. O padrão provisório continua `gemma4:e4b-it-qat`, e o padrão definitivo continua dependendo do benchmark de [MODELS.md §5](../MODELS.md).

## Consequências
- ~18 GB a mais no SSD (`OLLAMA_MODELS`).
- A tag Thinking instalada não é usada pelo dublador.
