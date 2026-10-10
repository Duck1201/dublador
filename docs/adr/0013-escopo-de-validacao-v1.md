# ADR-0013 — V1 validada com vídeos curtos, suporte até 1 h

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
O tempo de processamento na RX 7600 é dominado pelo TTS com clonagem (estimativa grosseira: 0,5x–2x o tempo real). Iterar o ajuste fino sobre vídeos longos é lento.

## Decisão
- **Alvo de validação da v1:** vídeos curtos (YouTube, aulas; até ~30 min; 1–3 falantes), com um vídeo de referência curto usado nos benchmarks.
- **Suporte documentado:** vídeos de até **1 hora**. Acima disso pode funcionar, mas não é validado.
- **Critério de sucesso:** terminar **sem intervenção**. Não há meta de velocidade.

## Consequências
- Retomada por estágio (ADR-0007) é obrigatória, não um extra.
- Escolhas de modelo priorizam qualidade dentro do que cabe em 8 GB de VRAM.
