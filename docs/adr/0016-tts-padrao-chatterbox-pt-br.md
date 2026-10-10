# ADR-0016 — TTS padrão: Chatterbox Multilingual V3 com finetune pt-BR

- **Status:** Substituído pela [ADR-0017](0017-tts-omnivoice-e-referencia-curta.md)
- **Data:** 2026-10-10

## Contexto
O primeiro TTS da pipeline, o Qwen3-TTS-12Hz-1.7B-Base, era o padrão provisório de [MODELS.md §6.2](../MODELS.md). Na primeira dublagem de ponta a ponta com voz real, o usuário avaliou a fala como **muito robotizada**. No mesmo job, o modelo não emitiu o token de fim em 2 de 4 segmentos ("Oi, pessoal." virou 655 s de áudio, 24 min de geração cada). O KV cache cresceu até ocupar os 8 GB da GPU, e o compositor do GNOME, que divide a placa, caiu por falta de VRAM.

## Decisão
O backend padrão do estágio `synthesize` passa a ser o **Chatterbox Multilingual V3 com o finetune `ResembleAI/Chatterbox-Multilingual-pt-br`**: o único finetune dedicado a PT-BR feito por um laboratório, com licença MIT. A escolha foi do usuário, entre as opções da shortlist do §6.2.

- O código vem do GitHub do Chatterbox, fixado num commit, porque o PyPI 0.1.7 é anterior ao checkpoint V3. O `torch` sobe para `2.11.0+rocm7.2` via `override-dependencies` (o pacote fixa a 2.6.0).
- O Qwen3-TTS sai do estágio: o `qwen-tts` fixa `transformers==4.57.3` e o Chatterbox fixa a `5.2.0`, então os dois não cabem no mesmo venv. O código continua no histórico do git, se o bake-off o trouxer de volta.
- **Independente de backend:** cada segmento tem um teto de geração (`max(4 s, 3 × janela, 0,2 s × caracteres)`). Uma geração que atinge o teto é refeita uma vez e, se falhar de novo, o áudio cortado é mantido com um warning no relatório. A pipeline não para.

## Consequências
- Pico de VRAM medido em **3,6 GB** (contra 5,0 GB do Qwen3-TTS), com mais folga para o desktop.
- A saída leva a marca d'água inaudível Perth, aplicada pelo próprio modelo.
- O Chatterbox só fala PT neste finetune: `languages.target` diferente de `pt-*` é rejeitado pelo backend.
- O bake-off do §6.2 continua valendo para a decisão definitiva; esta ADR troca só o padrão provisório.
