# ADR-0001 — Escopo: EN→PT-BR, idiomas parametrizados

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
O caso de uso é traduzir vídeos estrangeiros para assistir em português. A qualidade de ASR, tradução e principalmente TTS com clonagem varia muito por idioma; suportar uma matriz de pares desde o início multiplica testes sem benefício imediato.

## Decisão
O alvo da v1 é **inglês → português brasileiro**. Os idiomas são parâmetros da CLI (`--source-lang`, `--target-lang`) e da config desde o início, mas só EN→PT-BR é otimizado e validado.

## Consequências
- Os modelos são escolhidos pelo desempenho em ASR inglês e TTS PT-BR.
- Outros pares podem funcionar, mas são documentados como "não validados".
- Nenhum código pode assumir o idioma fixo (prompts, tokenização de legendas etc. leem a config).
