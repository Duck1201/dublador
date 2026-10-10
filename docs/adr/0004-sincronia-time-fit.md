# ADR-0004 — Sincronia por segmento (time-fit), sem lip-sync

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
Português costuma ser 20–30% mais longo que inglês; a fala dublada tende a estourar a janela da original e invadir a próxima. Lip-sync (alterar a boca no vídeo) resolveria o visual, mas é pesado, provavelmente não cabe nos 8 GB junto ao resto e tem suporte ruim em ROCm.

## Decisão
Cada fala dublada deve caber na janela de tempo da fala original. Estratégia em ordem:
1. traduzir já com orçamento de duração/caracteres;
2. *time-stretch* com preservação de pitch até um limite (padrão 1,25x);
3. usar o silêncio após a fala;
4. se ainda estourar, pedir ao LLM uma versão mais curta e ressintetizar só esses segmentos (até `max_fit_iterations`);
5. em último caso, acelerar além do limite e registrar aviso — a pipeline não para.

Lip-sync fica documentado como fase futura e opcional.

## Consequências
- O tradutor precisa aceitar restrição de tamanho e um modo `shorten` (ADR-0005).
- O orquestrador implementa um loop `translate → synthesize → fit` restrito aos segmentos em overflow.
- Os lábios não batem; o *timing* bate.
