# ADR-0008 — Revisão humana opcional após a tradução

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
A diarização pode juntar ou dividir falantes, o classificador de gênero pode errar e o LLM pode traduzir mal. Descobrir isso só no vídeo final desperdiça o estágio mais lento (TTS).

## Decisão
Com `--review`, a pipeline pausa **após a tradução**, gera um `review.yaml` legível (falantes, gênero, voz escolhida, texto EN e PT-BR por segmento) e o abre no `$EDITOR`. As correções são aplicadas e a pipeline continua com `dublador resume`. Sem a flag, roda de ponta a ponta.

## Consequências
- O `review.yaml` precisa de mapeamento bidirecional e validação contra os JSONs do contrato.
- Edições invalidam (via hash) os estágios seguintes no manifest.
