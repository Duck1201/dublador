# ADR-0005 — Tradução por LLM local configurável

- **Status:** Aceito; opção de qualidade ajustada pela [ADR-0015](0015-tradutor-qualidade-instruct.md)
- **Data:** 2026-10-09

## Contexto
O time-fit (ADR-0004) exige um tradutor capaz de encurtar falas mantendo o sentido. Modelos de tradução dedicados (NLLB, MarianMT) traduzem frase a frase, sem contexto e sem controle de tamanho. APIs na nuvem quebrariam o requisito de pipeline local. A máquina já tem Ollama com `gemma4:e4b-it-qat` (6,1 GB, cabe inteiro na VRAM) e `qwen3:30b-a3b` (18 GB MoE, roda com offload para RAM).

## Decisão
A tradução é feita por um **LLM local servido pelo Ollama**, recebendo janelas do diálogo com contexto, falantes e duração-alvo por segmento, e respondendo em JSON validado contra schema. O modelo é configurável (`--translator-model`). Padrão **provisório**: `gemma4:e4b-it-qat`; opção de qualidade: `qwen3:30b-a3b` (com raciocínio desativado). O padrão definitivo será fixado por um benchmark comparativo descrito em `docs/MODELS.md`.

## Consequências
- Nenhum modelo de tradução novo precisa ser baixado.
- A saída do LLM precisa de validação e retentativa (JSON inválido, segmentos faltando, alucinação).
- O Ollama precisa liberar a VRAM antes do estágio seguinte (`keep_alive: 0` ao final do estágio).
