# ADR-0003 — Separar voz do fundo em vez de descartar o áudio original

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
O plano original diz que o áudio original "será removido". Removido por inteiro, o vídeo final perde música, efeitos e ambiente, e fica artificial.

## Decisão
"Remover o áudio original" passa a significar **separar a voz do restante** com um modelo de *source separation* (classe Demucs). A trilha `vocals` alimenta transcrição, diarização e as referências de clonagem; a trilha `accompaniment` é preservada e mixada de volta com a dublagem.

## Consequências
- Mais um modelo na pipeline (cabe em 8 GB de VRAM).
- ASR e clonagem melhoram por trabalharem sobre voz limpa.
- Resíduos de voz original no `accompaniment` podem vazar na mixagem final; tratado no estágio `mix`.
- O modo voice-over/ducking (manter o áudio original baixo sob a dublagem) fica **fora do escopo** e registrado como ideia futura.
