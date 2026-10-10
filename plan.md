# Plano do projeto — dublador

> Documento vivo: resume **o que** vamos construir e **em que ordem**. O detalhe técnico está em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); as decisões e seus motivos, em [`docs/adr/`](docs/adr/).

## Objetivo

Criar uma pipeline de IA **100% local** que dubla vídeos de **inglês para português brasileiro**, operada por uma CLI Linux, preservando a voz de cada ator e a trilha sonora original.

## Fluxo

O usuário escolhe um vídeo na CLI (`dublador run video.mp4`, ou um seletor `fzf` sem argumento), e a pipeline:

1. extrai o áudio do vídeo;
2. **separa a voz do fundo** — a voz original é removida, mas música, efeitos e ambiente são preservados ([ADR-0003](docs/adr/0003-separacao-voz-fundo.md));
3. transcreve as falas em inglês e identifica **quem fala quando** (diarização);
4. identifica cada **falante**: gênero e um trecho limpo da voz para clonagem;
5. traduz para PT-BR com um LLM local, com contexto do diálogo e o tempo disponível para cada fala — *(opcional: pausa para revisão humana, `--review`)*;
6. gera a fala em PT-BR **clonando a voz de cada ator**, ou com uma voz de banco por gênero quando não há áudio limpo suficiente;
7. ajusta o tempo de cada fala para caber na janela original;
8. mixa a dublagem com o fundo original e gera um **MKV** com áudio PT-BR + áudio EN original e legendas PT-BR/EN, sem reencodar o vídeo.

## Requisitos-chave

- **Um falante = uma voz.** A pipeline precisa distinguir cada ator e saber seu gênero, para que o TTS gere uma voz consistente por ator: clonagem por padrão, banco de vozes por gênero como fallback ([ADR-0002](docs/adr/0002-vozes-hibridas.md)).
- **Sincronia por fala**, sem lip-sync: a fala dublada cabe na janela da fala original ([ADR-0004](docs/adr/0004-sincronia-time-fit.md)).
- **Roda na máquina de desenvolvimento:** Ryzen 5 5600, 32 GB de RAM, Radeon RX 7600 de 8 GB (ROCm). Um modelo por vez na VRAM ([ADR-0007](docs/adr/0007-venv-por-estagio-e-contrato-de-arquivos.md)).
- **Termina sem intervenção** em vídeos de até 1 hora; não há meta de velocidade ([ADR-0013](docs/adr/0013-escopo-de-validacao-v1.md)).
- **Retomável:** se algo falhar, continua do ponto em que parou (`dublador resume`).
- **Uso pessoal.** O código é público (Apache-2.0); os modelos são baixados pelo usuário sob as licenças deles ([ADR-0006](docs/adr/0006-uso-pessoal-e-licencas.md)).

## Fora do escopo da v1

Lip-sync, modo voice-over/ducking, processamento em lote/fila, Docker, outros pares de idiomas validados (os parâmetros existem, mas só EN→PT-BR é testado) e vídeos acima de 1 h.

## Sequência de implantação

### 1. Levantamento de como esse tipo de aplicação funciona — ✅ concluído

Resultado: [`docs/RESEARCH.md`](docs/RESEARCH.md). Nenhum achado contradiz a arquitetura; os ajustes recomendados (controle de tamanho da tradução por código, ancoragem por VAD, atenuação do resíduo de voz no fundo, cache por segmento) já foram incorporados ao [`ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### 2. Escolha dos modelos locais para a máquina — 🔄 em andamento

Resultado provisório: [`docs/MODELS.md`](docs/MODELS.md), com um padrão por estágio. Para fechar esta etapa, falta:

- [ ] **Preparar o ambiente no SSD**: cache do `uv`, venvs e modelos ([ADR-0014](docs/adr/0014-tudo-no-ssd.md)). → [#8](https://github.com/Duck1201/dublador/issues/8)
- [ ] **Validar o ROCm** com o `torch` ROCm na RX 7600, sem `HSA_OVERRIDE_GFX_VERSION`. → [#10](https://github.com/Duck1201/dublador/issues/10)
- [ ] **Decidir o motor de ASR**: WhisperX na GPU (CTranslate2 ROCm), WhisperX na CPU ou Qwen3-ASR. → [#13](https://github.com/Duck1201/dublador/issues/13)
- [ ] **Comparar os TTS** (a etapa mais aberta): a pesquisa ampliada está concluída e a shortlist tem 6 modelos ([MODELS.md §6.2](docs/MODELS.md#62-shortlist-do-bake-off-provisória)). Falta o teste comparativo com as mesmas frases PT-BR e referência EN e PT, avaliando sotaque (teste cego), similaridade de voz, inteligibilidade, VRAM e controle de duração.
- [ ] **Rodar o benchmark do tradutor**: `gemma4:e4b-it-qat` contra `qwen3:30b-a3b-instruct-2507-q4_K_M` ([ADR-0015](docs/adr/0015-tradutor-qualidade-instruct.md)).
- [ ] **Escolher o vídeo de referência** curto (5–10 min, 2–3 falantes, com música), que será usado em todos os testes. → [#11](https://github.com/Duck1201/dublador/issues/11)

A lista completa de testes está em [MODELS.md › Tarefas de validação](docs/MODELS.md#tarefas-de-validação). Esta etapa termina quando cada estágio tiver um modelo **testado na máquina**, e não só pesquisado.

### 3. Implementação — 🔄 planejada, roda em paralelo à etapa 2

Em marcos incrementais. Cada marco termina com o vídeo de referência sendo processado de ponta a ponta. O M0 não depende de modelos e começa já; o M1 usa os defaults provisórios do [`MODELS.md`](docs/MODELS.md) atrás de backends trocáveis, então o bake-off de TTS e a decisão de ASR trocam só o backend do estágio.

Os tickets ficam no GitHub: mapa do [M0 (#1)](https://github.com/Duck1201/dublador/issues/1), mapa do [M1 (#2)](https://github.com/Duck1201/dublador/issues/2) e resumo de [M2–M6 (#3)](https://github.com/Duck1201/dublador/issues/3), que serão detalhados quando o marco anterior fechar.

| Marco | Entrega | Pronto quando |
|---|---|---|
| **M0 — Fundação** | Esqueleto do repositório, `config.toml`, `dublador doctor` e `dublador setup`, contrato de arquivos e `manifest.json` | `doctor` aponta GPU, ffmpeg, Ollama e espaço em disco corretamente |
| **M1 — Esqueleto ponta a ponta** | `extract → separate → transcribe → speakers (v0) → translate → synthesize → mix → mux`, com **uma única voz** (clonada de um trecho de referência, sem diarização) e sem ajuste de tempo | O vídeo de referência sai como MKV dublado, ainda que fora de sincronia |
| **M2 — Falantes** | `diarize` + `speakers` com vários falantes (gênero por falante, trechos de referência), clonagem com fallback para o banco por gênero | Cada ator tem uma voz própria e consistente |
| **M3 — Sincronia** | Orçamento em sílabas, N candidatos de tradução, `fit` (VAD, time-stretch, silêncio) e loop `shorten` | Nenhuma fala invade a seguinte; `report.md` lista os overflows |
| **M4 — Mixagem e saída** | Atenuação do resíduo de voz, volume por clipe, loudness/limiter, legendas EN e PT-BR | Sem inglês audível por baixo da dublagem; legendas sincronizadas |
| **M5 — Revisão e retomada** | `--review` com `review.yaml`, cache por segmento, `resume`, `status`, `clean` | Editar uma fala e retomar refaz só aquela fala |
| **M6 — Validação v1** | Vídeo de ~1 h processado sem intervenção | Critérios da [ADR-0013](docs/adr/0013-escopo-de-validacao-v1.md) atendidos |

## Riscos abertos

| Risco | Impacto | Mitigação |
|---|---|---|
| Nenhum TTS local clona bem do inglês para PT-BR sem sotaque | Dublagem soa estrangeira | Teste comparativo com critério de sotaque; o TTS é plugin trocável; fallback para o banco de vozes |
| CTranslate2/WhisperX não roda na GPU AMD | ASR mais lento | ASR na CPU (int8) ou Qwen3-ASR em PyTorch |
| Instabilidade do ROCm na gfx1102 (placa não suportada oficialmente) | Travamentos em jobs longos | Teste de estresse de 30 min; override `11.0.0` como plano B; retomada por estágio |
| Português ~20–30% mais longo que inglês | Falas aceleradas demais | Controle de tamanho por código + loop `shorten` + TTS com duração-alvo |
| Espaço em disco | Instalação falha | Tudo no SSD ([ADR-0014](docs/adr/0014-tudo-no-ssd.md)); `dublador clean` |
