# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado do projeto

`dublador` é uma pipeline de IA 100% local que dubla vídeos EN→PT-BR por uma CLI Linux. A implementação está no marco **M0 (Fundação)**. Já funcionam a config (`config.py`), o contrato (`packages/stagekit`), o job e o `manifest.json` (`jobs.py`, `manifest.py`), o runner com retomada por hash de entradas (`runner.py`, registro em `stages.py`) e os estágios ffmpeg `extract`/`mux` (`ffmpeg.py`), com `run`, `resume`, `status`, `setup` (venvs por estágio fora do repositório, `setup.py`) e `doctor` (`doctor.py`). No M1, os estágios `separate`, `transcribe`, `speakers` (v0), `translate`, `synthesize` e `mix` já existem em `stages/`, cada um com venv próprio em `<venvs_dir>/<nome>`, e `dublador run` gera o MKV dublado de ponta a ponta (sem ajuste de tempo, que é o M3). No M2, o estágio `diarize` (pyannote community-1, modelo de acesso restrito no HF: login uma vez com `HF_HOME=<models_dir>/hf`) separa os falantes, o `speakers` classifica o gênero (ECAPA na CPU) e escolhe uma voz do banco (`voices/README.md`, gerado pelo `setup`) para quem não tem referência, e o `translate` recebe o gênero de quem fala e de quem ouve. O `transcribe` roda o Whisper na GPU com o wheel ROCm do CTranslate2, que o `dublador setup` baixa e liga em `stages/transcribe/wheels/` (ignorado pelo git) antes do `uv sync`; sem esse link, o `uv sync` do estágio falha. `clean` ainda sai com "not implemented yet". Para rodar os testes de um estágio: `UV_PROJECT_ENVIRONMENT=<venvs_dir>/<nome> uv run --project stages/<nome> pytest stages/<nome>` — nunca deixe `UV_PROJECT_ENVIRONMENT` exportada ao rodar comandos `uv` na raiz, senão o `uv` recria o venv do estágio com o projeto do orquestrador. O `config.toml` local desta máquina aponta para `~/SSD/Extras/dublador`. Os testes de orquestração usam estágios falsos em `tests/fake_stages/` e um vídeo sintético gerado com ffmpeg. O plano e os tickets estão no `plan.md` e nos issues do GitHub (mapas: #21 M2, #29 M3, #3 M4–M6).

## Comandos

```bash
uv sync                      # cria o .venv do orquestrador (Python 3.13)
uv run dublador --help
uv run ruff check .          # lint
uv run ruff format .         # formatação (use --check no CI)
uv run pyright               # tipos
uv run pytest                # testes; um só: uv run pytest tests/test_cli.py::test_version
```

O CI (`.github/workflows/ci.yml`) roda essas quatro checagens, sem GPU. O `pytest` da raiz também roda os testes de `packages/stagekit` (o contrato compartilhado, instalado como dependência editável); ele precisa continuar compatível com Python 3.10+. O `ruff.toml` da raiz mira o Python 3.13; subprojetos que rodam em Python mais antigo (como `packages/stagekit`, 3.10+) têm um `ruff.toml` que estende o da raiz e baixa o `target-version`.

Leia nesta ordem antes de implementar qualquer coisa:
1. `docs/ARCHITECTURE.md`: estágios, contrato de arquivos, CLI, configuração e layout proposto do repositório.
2. `docs/adr/README.md`: índice das decisões. As ADRs aceitas são a fonte de verdade sobre o que foi decidido com o usuário.
3. `docs/MODELS.md`: modelo padrão por estágio, ambiente ROCm, VRAM e o checklist "Tarefas de validação". Itens marcados [NV] não foram verificados.
4. `docs/RESEARCH.md`: técnicas e armadilhas. A §5 mapeia os achados para a arquitetura.

## Arquitetura em uma página

- **Orquestrador + estágios isolados.** O pacote `dublador` (CLI Typer) chama cada estágio (`separate`, `transcribe`, `diarize`, `speakers`, `translate`, `synthesize`, `fit`, `mix`) como **subprocesso**, cada um com o próprio `pyproject.toml` e venv `uv`. Os estágios `extract` e `mux` são só `ffmpeg`, rodando dentro do orquestrador. Os estágios **nunca** importam código uns dos outros: eles se comunicam por arquivos JSON (com schema versionado, ex.: `"schema": "transcript/v1"`) e WAV no diretório do job, sempre com escrita atômica.
- **Um modelo por vez na VRAM**, porque a GPU é uma RX 7600 de 8 GB. Os estágios rodam em sequência e cada um sai ao terminar, liberando a GPU. O estágio de tradução usa o Ollama e precisa enviar `keep_alive: 0` ao terminar, senão o TTS seguinte fica sem memória.
- **Retomada:** o `manifest.json` do job guarda o status e o hash das entradas por estágio. Dentro de cada estágio, o cache é por segmento. A interface padrão de um estágio é `python -m <stage> --job-dir ... --config ... [--only seg_0007,...]`, com progresso em JSON lines no stdout.
- **Loop de ajuste de tempo:** os segmentos que estouram a janela da fala original voltam para `translate` (modo `shorten`), `synthesize` e `fit`, só pelos IDs afetados. A pipeline nunca para por overflow.
- **O backend de TTS é um plugin trocável.** O modelo de TTS ainda não está fixado e depende de um teste comparativo.

## Convenções

- **Idioma** (ADR-0012): a documentação (README, `docs/`, ADRs, este arquivo) fica em PT-BR. O código, os comentários, os commits e as mensagens de CLI e de erro ficam em inglês.
- **ADRs:** uma ADR aceita não é editada. Se uma decisão mudar, crie uma nova ADR, marque a antiga como `Substituído pela ADR-XXXX` e atualize o índice em `docs/adr/README.md`. Mudanças de escopo são decisões do usuário e precisam ser confirmadas com ele antes.
- **Nenhum caminho pessoal no código.** Os caminhos vêm de `config.toml` (o modelo versionado é `config.example.toml`; o `config.toml` fica fora do git) ou de variáveis de ambiente (`DUBLADOR_WORK_DIR`, `DUBLADOR_MODELS_DIR`, `DUBLADOR_VENVS_DIR`, `UV_CACHE_DIR`).
- **Disco** (ADR-0014): o `/home` tem pouco espaço livre. O cache do `uv`, os venvs, os modelos e os arquivos de trabalho ficam no SSD (`~/SSD/Extras/dublador/` nesta máquina), com cache e venvs no mesmo filesystem para aproveitar hardlink/reflink. Não instale `torch` ROCm nem baixe modelos para o `/home`.
- **Nunca commitar** pesos de modelos, vídeos ou áudios (o `.gitignore` já cobre isso). O código é Apache-2.0; os modelos têm licenças próprias, algumas não comerciais, registradas em `docs/MODELS.md`.

## Ambiente da máquina de desenvolvimento

- Ryzen 5 5600, 32 GB de RAM, Radeon RX 7600 de 8 GB (gfx1102). O Python do sistema é o 3.13; a versão de Python é escolhida por estágio.
- Linha de base do `torch`: `2.11.0+rocm7.2`. Alguns estágios têm pins rígidos (o WhisperX pede `torch~=2.8`), e por isso existem venvs separados.
- Comece **sem** `HSA_OVERRIDE_GFX_VERSION`, porque os wheels do PyTorch já trazem kernels gfx1102. Use `11.0.0` só se alguma biblioteca falhar com arquitetura não suportada.
- O Ollama roda como serviço com backend **Vulkan** e modelos no SSD. Tradutor padrão provisório: `gemma4:e4b-it-qat`. Opção de qualidade: `qwen3:30b-a3b-instruct-2507-q4_K_M`. A tag `qwen3:30b-a3b` instalada é a variante Thinking e não deve ser usada.
- Não existe `/opt/rocm`; os pacotes ROCm são do Debian e os wheels do PyTorch trazem o próprio runtime.

## Agent skills

### Issue tracker

Issues e specs ficam no GitHub Issues de `Duck1201/dublador` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Labels padrão: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` na raiz (criado sob demanda) + `docs/adr/`. See `docs/agents/domain.md`.
