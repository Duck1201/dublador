# ADR-0018 — Ordem do ajuste de tempo e loop de encurtamento

- **Status:** Aceito (detalha a [ADR-0004](0004-sincronia-time-fit.md))
- **Data:** 2026-10-10

## Contexto
A ADR-0004 fixou a estratégia de sincronia por segmento, mas deixou em aberto a ordem exata dos passos, o papel da duração-alvo do TTS e como o loop sabe o que pedir a cada estágio. Desde então mudaram três coisas:

- o TTS passou a ser o OmniVoice, que aceita a duração em segundos (ADR-0017);
- a calibração no vídeo de referência mostrou fala estável em torno de 5,7 sílabas/s;
- o M1 mostrou falas, na mediana, 15% mais longas que a janela original, chegando a 2,9×.

O usuário aprovou o plano do M3 com a ordem abaixo.

## Decisão
**Na tradução:** cada fala recebe um orçamento em **sílabas**, calculado sobre a janela do seu início até o início da próxima fala (de qualquer falante), menos 150 ms, vezes `translate.syllables_per_second`. O LLM devolve **3 candidatos**, e o código escolhe o mais longo que cabe; se nenhum couber, escolhe o mais curto.

**No estágio `fit`, para cada segmento:**
1. **fits:** o áudio cabe na duração da fala original;
2. **silence:** cabe na pausa que vem depois, até o início da próxima fala menos `fit.gap_seconds`;
3. **stretch:** time-stretch com pitch preservado (`rubberband` do ffmpeg) até `fit.max_speedup` (1,25×);
4. **overflow:** acelera até `fit.max_forced_speedup` (1,6×), para o mix sempre ter áudio, e pede ajuda ao loop.

**O loop do orquestrador** repete o ciclo para os ids em overflow, até `fit.max_fit_iterations` (2):
- **1º overflow seguido:** o `fit` grava `shorten_to` (meta de sílabas). O `translate --only` reescreve o **texto atual** mais curto, sem retraduzir do zero; depois vêm o `synthesize` e o `fit`.
- **2º overflow seguido:** o `fit` grava `target_duration`. O `synthesize --only` pede ao TTS essa duração; o valor fica gravado no `synthesis.json` e continua valendo enquanto texto, voz e backend não mudarem.
- **O que ainda estourar** fica com a aceleração forçada e um aviso no `report.md`. A pipeline não para.

**Âncora:** o início da fala é o início da primeira palavra alinhada pelo wav2vec2 (WhisperX), que já é preciso. Não há um VAD separado.

## Consequências
- A ordem difere da ADR-0004 em dois pontos:
  - o silêncio seguinte vem antes do stretch, porque esticar é pior que usar uma pausa livre;
  - a duração-alvo do TTS entra como segunda tentativa do loop, porque encurtar o texto preserva a naturalidade.
- `fit/v1` registra a estratégia, o fator de aceleração e as dicas de cada segmento, e o `mix` lê o `fit.json`.
- O manifest regrava os hashes de entrada depois do loop, para o `resume` não refazer nada.
- O resumo e o glossário do vídeo, previstos na arquitetura, continuam fora (M4+).
