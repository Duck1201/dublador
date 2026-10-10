# ADR-0007 — Um venv por estágio, contrato de arquivos, execução sequencial

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
As bibliotecas de separação, ASR, diarização e TTS costumam fixar versões conflitantes de `torch`, `numpy` e `transformers`, várias ainda não suportam o Python 3.13 do sistema, e todas precisam do `torch` com build ROCm. A GPU tem 8 GB, insuficiente para manter mais de um modelo grande carregado. Docker (imagens `rocm/pytorch` com mais de 15 GB) não cabe bem no disco e adiciona atrito com a GPU AMD.

## Decisão
- Cada estágio é um **subprojeto Python com venv `uv` próprio** (versão de Python escolhida por estágio), invocado pelo orquestrador como **subprocesso**.
- Os estágios se comunicam **apenas por arquivos** no diretório do job (JSON com schema versionado + WAV), com escrita atômica.
- Os estágios rodam **em sequência**: cada um carrega seu modelo, processa o vídeo inteiro e sai, liberando a VRAM.
- Um `manifest.json` por job registra estado e hash das entradas, permitindo **retomar** a partir do primeiro estágio incompleto ou alterado.

## Consequências
- Trocar um modelo/biblioteca afeta só um diretório.
- Retomada e revisão humana (ADR-0008) saem quase de graça.
- Mais disco para venvs; mitigado mantendo os venvs no mesmo filesystem do cache do `uv` (hardlinks) — a validar.
- Custo de carregar modelos uma vez por estágio (aceitável; não há meta de velocidade).
