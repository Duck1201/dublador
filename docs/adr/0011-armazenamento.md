# ADR-0011 — Código em /home, modelos e trabalho no SSD, tudo configurável

- **Status:** Substituído pela [ADR-0014](0014-tudo-no-ssd.md)
- **Data:** 2026-10-09

## Contexto
A máquina de desenvolvimento tem só ~32 GB livres em `/home` (NVMe, 92% ocupado), com 18 GB já usados pelo cache do `uv`, e ~230 GB livres em um SSD montado em `~/SSD/Extras`. Modelos novos somam ~6–10 GB e cada vídeo de 1 h gera vários GB de WAVs intermediários. O repositório é público, então caminhos pessoais não podem estar no código.

## Decisão
- Código e venvs dos estágios ficam no repositório (em `/home`), no mesmo filesystem do cache do `uv`, para aproveitar hardlinks.
- Pesos de modelos (`models_dir`) e diretórios de trabalho (`work_dir`) ficam no SSD por padrão local (`~/SSD/Extras/dublador/{models,work}`).
- Tudo é configurável via `config.toml` e variáveis de ambiente (`DUBLADOR_WORK_DIR`, `DUBLADOR_MODELS_DIR`); o orquestrador aponta `HF_HOME`/`TORCH_HOME` dos estágios para `models_dir`.
- `dublador clean` remove intermediários pesados de jobs concluídos, mantendo os JSONs.

## Consequências
- O `config.example.toml` versionado usa caminhos genéricos; o `config.toml` local fica fora do git.
- `dublador doctor` verifica espaço livre antes de rodar.
