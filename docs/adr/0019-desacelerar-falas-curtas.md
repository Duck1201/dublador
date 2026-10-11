# ADR-0019 — Desacelerar levemente as falas que ficam curtas

- **Status:** Aceito (ajusta a [ADR-0018](0018-ordem-do-time-fit-e-loop.md))
- **Data:** 2026-10-10

## Contexto
No aceite do M2 + M3, o usuário ouviu "partes sem áudio". A análise do vídeo de referência mostrou 8 trechos, de 1,5 a 2,5 s cada (14 s em 10 min), em que a fala em português termina antes de o palestrante parar de falar na tela. O OmniVoice fala mais rápido que esse palestrante, que é pausado, e a ADR-0018 só acelerava: uma fala curta ficava como estava ("fits") e deixava silêncio no fim.

## Decisão
No estágio `fit`, uma fala mais curta que a original é **desacelerada com pitch preservado** (`rubberband`) em direção à duração original, sem passar de `fit.min_speedup` (padrão **0,85×**), dentro da faixa 0,80–1,25× que o [MODELS.md §8](../MODELS.md) considera natural. Ela continua marcada como "fits", com `ratio` < 1 no `fit.json`.

## Consequências
- O silêncio no fim das falas curtas diminui (por exemplo, 4,3 s de fala para 6 s de original passa a ocupar 5,1 s), mas não some quando o português é muito mais curto. Encher o resto exigiria uma tradução mais longa ou a duração-alvo do TTS, o que fica para depois se o usuário ainda sentir falta.
- `fit.min_speedup = 1.0` desliga o ajuste.
