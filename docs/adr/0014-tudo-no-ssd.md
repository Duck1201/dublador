# ADR-0014 — Cache do uv, venvs e modelos no SSD

- **Status:** Aceito (substitui a [ADR-0011](0011-armazenamento.md))
- **Data:** 2026-10-09

## Contexto
A pesquisa de modelos ([MODELS.md](../MODELS.md)) mostrou que o wheel do `torch` ROCm tem ~6 GB compactado e que alguns estágios precisam de versões diferentes do `torch` (ex.: 2.11 como base, 2.8 para WhisperX, possivelmente 2.6 para Chatterbox). Cada versão ocupa o cache do `uv`, que já tem 18 GB no `/home`, onde restam ~32 GB livres. Manter os venvs no `/home` (ADR-0011) enche o disco. Hardlink e reflink só funcionam se cache e venvs ficarem no mesmo filesystem.

## Decisão
- **Só o código** fica no `/home` (o repositório).
- **Cache do `uv`** (`UV_CACHE_DIR`), **venvs dos estágios**, **pesos de modelos** e **diretórios de trabalho** ficam no SSD, por padrão local em `~/SSD/Extras/dublador/{uv-cache,venvs,models,work}` (btrfs, com hardlink/reflink entre cache e venvs).
- Os venvs dos estágios ficam em `<venvs_dir>/<estágio>` (via `UV_PROJECT_ENVIRONMENT`), e não mais em `stages/<nome>/.venv`.
- O restante da ADR-0011 continua valendo: tudo configurável via `config.toml` e variáveis de ambiente, nenhum caminho pessoal no código, e `dublador clean`.
- Os modelos do Ollama já ficam no SSD (`OLLAMA_MODELS` do serviço), fora do controle do dublador.

## Consequências
- Nova chave de config `paths.venvs_dir` e variáveis de ambiente `DUBLADOR_VENVS_DIR` e `UV_CACHE_DIR`; o `dublador setup` cria os venvs lá.
- O `/home` não cresce com o uso do projeto.
