# Arquitetura do dublador

Este documento descreve **como** a pipeline é organizada: estágios, contrato de arquivos entre eles, orquestração e configuração. A escolha concreta de cada modelo está em [`MODELS.md`](MODELS.md); o porquê de cada decisão está nos [ADRs](adr/).

## Visão geral

```
                     ┌──────────────────── orquestrador (CLI `dublador`) ────────────────────┐
                     │  lê config, cria o job, chama cada estágio como subprocesso, controla │
                     │  manifest.json, retomada (resume), pausa de revisão e loop de ajuste  │
                     └───────────────────────────────────────────────────────────────────────┘
                                                     │
 video.mp4                                           ▼
   │   01 extract ──► 02 separate ──► 03 transcribe ──► 04 diarize ──► 05 speakers
   │       │              │  vocals         │                              │
   │       │              └─ accompaniment ─┼──────────────────────┐       │
   │       │                                ▼                      │       ▼
   │       │                          06 translate ◄──── shorten ──┤  [--review]
   │       │                                │                      │       │
   │       │                                ▼                      │       │
   │       │                          07 synthesize ──► 08 fit ────┘       │
   │       │                                                │              │
   │       │                                                ▼              │
   │       └──────────── áudio EN original ───────────► 09 mix ◄───────────┘
   │                                                        │
   └──────────────────── stream de vídeo (copy) ──────► 10 mux ──► video.ptbr.mkv
```

Princípios:

1. **Um estágio = um processo = um venv.** Cada estágio é um subprojeto Python isolado com o próprio `pyproject.toml` e venv `uv`. Os estágios não importam código uns dos outros; falam apenas por arquivos. ([ADR-0007](adr/0007-venv-por-estagio-e-contrato-de-arquivos.md))
2. **Um modelo por vez na VRAM.** A RX 7600 tem 8 GB. Cada estágio carrega o seu modelo, processa o vídeo inteiro, grava a saída e sai, liberando a GPU. Nunca há dois estágios rodando ao mesmo tempo.
3. **Tudo é retomável.** A saída de cada estágio é persistida no diretório do job. Se algo falhar, `dublador resume` continua do primeiro estágio incompleto.
4. **Arquivos legíveis por humanos.** Os artefatos intermediários são JSON/YAML e WAV; dá para inspecionar e editar qualquer etapa.

## Estágios

| # | Estágio | Entrada | Saída | Usa GPU | Venv |
|---|---|---|---|---|---|
| 01 | `extract` | vídeo | `01_extract/audio.wav`, `01_extract/probe.json` | não | orquestrador (só `ffmpeg`) |
| 02 | `separate` | `audio.wav` | `02_separate/vocals.wav`, `02_separate/accompaniment.wav` | sim | `stages/separate` |
| 03 | `transcribe` | `vocals.wav` | `03_transcribe/transcript.json` (segmentos + palavras com timestamps) | sim | `stages/transcribe` |
| 04 | `diarize` | `vocals.wav`, `transcript.json` | `04_diarize/diarization.json` (turnos por falante) + `transcript.json` com `speaker` por segmento | sim | `stages/diarize` |
| 05 | `speakers` | `vocals.wav`, diarização | `05_speakers/speakers.json` (gênero, tempo de fala, qualidade) + `05_speakers/refs/<speaker>.wav` | sim | `stages/speakers` |
| 06 | `translate` | transcript + speakers | `06_translate/translation.json` | via Ollama | `stages/translate` |
| 07 | `synthesize` | translation + refs | `07_synthesize/segments/<id>.wav` + `synthesis.json` | sim | `stages/synthesize` |
| 08 | `fit` | segmentos sintetizados | `08_fit/segments/<id>.wav` + `fit.json` (estratégia aplicada, overflow) | não | `stages/fit` |
| 09 | `mix` | segmentos ajustados + `accompaniment.wav` | `09_mix/dub.wav` | não | `stages/mix` |
| 10 | `mux` | vídeo + `dub.wav` + `audio.wav` + legendas | `<saída>.mkv`, `10_mux/en.srt`, `10_mux/pt-BR.srt` | não | orquestrador (só `ffmpeg`) |

Os modelos de cada estágio estão em [`MODELS.md`](MODELS.md). Os estágios 01 e 10 são só `ffmpeg` e rodam no próprio orquestrador.

### 01 — extract
Extrai a faixa de áudio principal em WAV (PCM, taxa e canais originais) e grava o `ffprobe` do arquivo (duração, streams, idioma das faixas). Se o vídeo tiver várias faixas de áudio, usa a marcada como padrão ou a escolhida por `--audio-track`.

### 02 — separate
Separa a voz (`vocals`) do restante (`accompaniment`: música, efeitos, ambiente). A trilha `vocals` alimenta os estágios 03–05; a `accompaniment` volta no estágio 09. ([ADR-0003](adr/0003-separacao-voz-fundo.md))

### 03 — transcribe
ASR em inglês sobre `vocals.wav`, com **timestamps por palavra** (necessários para alinhar falantes e montar legendas). Remove segmentos suspeitos de alucinação (texto em trechos de silêncio, repetições).

### 04 — diarize
Determina **quem fala quando** com o pyannote `speaker-diarization-community-1`, na versão sem sobreposição de fala (*exclusive*). Cada **palavra** vai para o turno que mais a cobre, e o segmento é cortado onde o falante muda; os ids são renumerados. Falantes com menos de `min_speaker_seconds` de fala são fundidos no vizinho mais próximo, e os rótulos viram `SPK_00`, `SPK_01`… por ordem de entrada. `num_speakers`, `min_speakers` e `max_speakers` (ou `dublador run --num-speakers`) forçam a contagem.

Saídas: `04_diarize/diarization.json` (`diarization/v1`) e `04_diarize/transcript.json`, o mesmo `transcript/v1` com `speaker` preenchido, que é o que o `speakers` e o `translate` leem.

O modelo é de acesso restrito no Hugging Face: aceite os termos e faça login uma vez (com o `HF_HOME` em `<models_dir>/hf`). O `dublador setup` baixa o modelo, e as execuções seguintes são offline. Na GPU, o caminho do MIOpen para RNNs fica desligado (`torch.backends.cudnn.enabled = False`), porque o MIOpen não compila o LSTM da segmentação na gfx1102.

### 05 — speakers
Para cada falante:
- calcula o tempo total de fala e escolhe o **melhor trecho de referência** para clonagem (3–4,5 s cortados em fronteira de palavra, [ADR-0017](adr/0017-tts-omnivoice-e-referencia-curta.md); fala contínua, sem sobreposição, sem música residual), guardando também a transcrição exata do trecho (alguns TTS a exigem);
- classifica o **gênero por falante** com o classificador ECAPA (`JaesungHuh/voice-gender-classifier`, na CPU) sobre até 5 trechos limpos de 2–6 s. O voto é ponderado pela duração e gera um nível de confiança; nunca se decide segmento a segmento. Abaixo de `min_gender_confidence`, um F0 mediano claro desempata; senão o gênero fica `unknown`;
- decide a estratégia de voz: `clone` se houver referência de pelo menos 2 s; senão `bank`, com `voice` = `<gênero>_NN` do banco de vozes ([`voices/README.md`](../voices/README.md)), estável por falante e sem repetir voz no mesmo vídeo. ([ADR-0002](adr/0002-vozes-hibridas.md))

### 06 — translate
Tradução EN→PT-BR por LLM local via Ollama. O LLM recebe janelas do diálogo (com contexto anterior/posterior e falantes) e, para cada segmento, a **duração disponível** e um orçamento aproximado de caracteres. Responde em JSON validado contra schema; respostas inválidas são re-solicitadas. O mesmo estágio tem um modo `shorten`, chamado pelo loop de ajuste (abaixo).

Como pedir uma duração só no prompt controla mal o tamanho do texto ([RESEARCH §3.5](RESEARCH.md)), o controle de comprimento é feito **por código**:

- o orçamento de cada segmento é expresso em **sílabas** (estimadas a partir da duração e de uma taxa de fala PT-BR), não só em segundos;
- o LLM gera **N candidatos** por segmento e o estágio escolhe o melhor que cabe no orçamento, contando as sílabas;
- um **resumo e um glossário do vídeo** (nomes, termos, tratamento você/senhor) são gerados numa primeira passada e entram no contexto de todas as janelas;
- o prompt inclui o **gênero de quem fala e de com quem fala**, para acertar a concordância em PT-BR. ([ADR-0005](adr/0005-traducao-por-llm-local.md))

**Ponto de revisão:** com `--review`, a pipeline para aqui, grava `review.yaml` (falantes, gênero, voz escolhida, texto EN e PT-BR por segmento) e o abre no `$EDITOR`. As edições são aplicadas de volta em `translation.json`/`speakers.json` e a pipeline segue com `dublador resume`. ([ADR-0008](adr/0008-revisao-opcional.md))

### 07 — synthesize
Gera um WAV por segmento com o TTS: clonagem zero-shot a partir de `refs/<speaker>.wav`, ou voz do banco no fallback. O backend de TTS é um **plugin trocável** — o estágio expõe uma interface fixa e o modelo por trás muda por configuração.

### 08 — fit (time-fit)
Faz cada fala dublada caber na janela da fala original ([ADR-0004](adr/0004-sincronia-time-fit.md)):

A janela de cada fala é ancorada no **início real da fala original** detectado por VAD sobre `vocals.wav` (o início do segmento do ASR costuma ser impreciso).

1. se cabe, mantém;
2. se estoura e o backend de TTS aceita **duração-alvo**, ressintetiza já na duração da janela (soa melhor do que esticar o áudio); depende do modelo escolhido em [`MODELS.md`](MODELS.md);
3. se estoura até o limite de aceleração (`fit.max_speedup`, padrão `1.25x`, configurável), aplica *time-stretch* com preservação de pitch;
4. se ainda estoura, usa o silêncio após a fala (até o início da próxima fala do mesmo falante ou de outro);
5. se ainda estoura, marca o segmento como **overflow**.

Como o vídeo é copiado sem reencodar ([ADR-0009](adr/0009-formato-de-saida.md)), não há a opção de desacelerar o vídeo; por isso o encurtamento da tradução é a principal alavanca.

### Loop de ajuste (orquestrador)
Segmentos marcados como overflow voltam ao `translate` em modo `shorten` (com a duração-alvo e o texto atual), depois a `synthesize` e `fit`, **só para esses IDs**, até `max_fit_iterations` (padrão 2). O que ainda estourar recebe aceleração acima do limite com aviso no log/relatório — a pipeline nunca para por isso (critério da v1: terminar sem intervenção).

### 09 — mix
Posiciona cada segmento no seu timestamp, soma com `accompaniment.wav` e grava `dub.wav`. Cuidados ([RESEARCH §3.8](RESEARCH.md)):

- **atenuação do fundo durante as falas**: o `accompaniment` quase sempre carrega resíduo da voz original em inglês; ele é abaixado alguns dB nos trechos com fala dublada. Isso é diferente do modo voice-over, que fica fora do escopo e mantém a voz original audível;
- o volume de cada clipe dublado é casado com o da fala original correspondente;
- sons não verbais curtos (risadas, gritos) que não foram transcritos são preservados a partir de `vocals.wav`;
- `loudnorm` e limiter no final.

### 10 — mux
Gera as legendas SRT (EN da transcrição com os tempos originais; PT-BR da tradução com os tempos do áudio **dublado** depois do fit; no máximo ~42 caracteres por linha) e monta a saída com `ffmpeg`, **sem reencodar o vídeo** (`-c:v copy`) ([ADR-0009](adr/0009-formato-de-saida.md)):

- MKV (padrão): faixa de áudio PT-BR (padrão) + faixa EN original + legendas PT-BR e EN;
- `--format mp4`: mesmas faixas, legendas como `mov_text`.

## Diretório do job (contrato de arquivos)

Cada vídeo processado tem um diretório de trabalho em `<work_dir>/<job_id>/`, onde `job_id` é derivado do nome do arquivo + hash curto do conteúdo:

```
<work_dir>/<job_id>/
├── manifest.json          # estado do job (ver abaixo)
├── config.snapshot.toml   # config efetiva usada no job
├── 01_extract/
├── 02_separate/
├── ...
├── 10_mux/
├── review.yaml            # só com --review
└── report.md              # resumo final: tempos, overflows, falantes, avisos
```

`manifest.json` registra, por estágio: status (`pending | running | done | failed`), horário de início/fim, versão do estágio, modelo usado e hash das entradas. Ao retomar, um estágio `done` cujas entradas não mudaram é pulado; se uma entrada mudou (por exemplo, `translation.json` editado na revisão), ele e os seguintes são refeitos.

Dentro de cada estágio, o cache também é **por segmento**: cada segmento carrega o hash das suas entradas (texto, voz, referência). Depois de uma edição na revisão, só os segmentos alterados são retraduzidos ou ressintetizados, e não o vídeo inteiro.

Regras do contrato:

- Um estágio só escreve **no próprio diretório**, gravando em arquivo temporário e renomeando no fim (escrita atômica) — um estágio interrompido nunca deixa saída "meio pronta".
- Os formatos JSON têm schema versionado (`"schema": "transcript/v1"`), validado na leitura e antes da escrita. Os schemas ficam em `packages/stagekit/src/stagekit/schemas/`.
- Tempos são sempre em **segundos (float) relativos ao início do vídeo**.
- IDs de segmento são estáveis entre estágios (`seg_0001`…).

## Interface de um estágio

Todo estágio é invocado da mesma forma pelo orquestrador:

```
<venv>/bin/python -m <stage> --job-dir <dir> --config <config.snapshot.toml> [--only seg_0007,seg_0012]
```

- `--only` restringe o processamento a alguns segmentos (usado pelo loop de ajuste).
- Progresso é emitido no stdout como **JSON lines** (`{"event": "progress", "done": 12, "total": 340}`), que o orquestrador transforma em barra de progresso.
- Código de saída `0` = sucesso; qualquer outro = falha, com a mensagem de erro em stderr.

Além do modo padrão, todo estágio tem dois subcomandos auxiliares:

```
<venv>/bin/python -m <stage> prefetch --config <config.toml>   # baixa os modelos para models_dir (usado por `dublador setup`)
<venv>/bin/python -m <stage> probe                             # imprime JSON `stage-probe/v1` com versões e GPU (usado por `dublador doctor`)
```

O parsing desses argumentos, a escrita atômica, a validação dos schemas e o emissor de progresso ficam no pacote compartilhado `packages/stagekit`. Ele contém só o contrato (nada de modelo), por isso não viola a regra de que um estágio não importa código de outro ([ADR-0007](adr/0007-venv-por-estagio-e-contrato-de-arquivos.md)).

## CLI

Construída com **Typer** ([ADR-0010](adr/0010-cli.md)). Mensagens em inglês.

```
dublador run [VIDEO] [--source-lang en] [--target-lang pt-BR]
             [--translator-model gemma4:e4b-it-qat] [--review]
             [--format mkv|mp4] [--output PATH] [--audio-track N]
dublador resume [JOB_ID | VIDEO]
dublador status [JOB_ID]
dublador clean [JOB_ID | --done] [--all]
dublador setup              # cria os venvs dos estágios e baixa os modelos
dublador doctor             # verifica ROCm, ffmpeg, Ollama, modelos e espaço em disco
```

Sem `VIDEO`, `run` abre um seletor com `fzf` (ou um prompt de caminho, se `fzf` não estiver instalado). O seletor só preenche argumentos; o núcleo da CLI é não interativo.

`dublador clean` remove os WAVs intermediários de jobs concluídos e mantém os JSONs (permitindo reprocessar só a partir da tradução, por exemplo).

## Configuração

Ordem de precedência: flags da CLI > variáveis de ambiente > `config.toml` > padrões.

```toml
# config.toml da máquina de desenvolvimento (o versionado é config.example.toml, com caminhos genéricos)
[paths]
work_dir   = "~/SSD/Extras/dublador/work"
models_dir = "~/SSD/Extras/dublador/models"
venvs_dir  = "~/SSD/Extras/dublador/venvs"

[languages]
source = "en"
target = "pt-BR"

[translate]
model = "gemma4:e4b-it-qat"
ollama_url = "http://localhost:11434"

[fit]
max_speedup = 1.25
max_fit_iterations = 2

[output]
format = "mkv"

[gpu]
# hsa_override_gfx_version = "11.0.0"   # só se alguma lib falhar com arch não suportada (ver MODELS.md)
```

| Variável | Efeito |
|---|---|
| `DUBLADOR_CONFIG` | caminho do `config.toml` |
| `DUBLADOR_WORK_DIR` | sobrescreve `paths.work_dir` |
| `DUBLADOR_MODELS_DIR` | sobrescreve `paths.models_dir` |
| `DUBLADOR_VENVS_DIR` | sobrescreve `paths.venvs_dir` |
| `UV_CACHE_DIR` | cache do `uv`; deve ficar no mesmo filesystem de `venvs_dir` (padrão local: `~/SSD/Extras/dublador/uv-cache`) |

O orquestrador exporta para cada estágio `HF_HOME`, `TORCH_HOME` etc. apontando para dentro de `models_dir`, e `HSA_OVERRIDE_GFX_VERSION` só se estiver definido na config. Nenhum caminho de usuário fica fixo no código. ([ADR-0014](adr/0014-tudo-no-ssd.md))

## Layout do repositório (proposto)

```
dublador/
├── README.md
├── LICENSE
├── config.example.toml
├── pyproject.toml            # orquestrador + CLI (pacote `dublador`)
├── src/dublador/             # CLI, orquestração, manifest, ffmpeg (01 e 10), loop de ajuste
├── packages/
│   └── stagekit/             # contrato compartilhado: args, config, escrita atômica, progresso e os
│                             # JSON Schemas (src/stagekit/schemas/), que vão junto para o venv de cada estágio
├── stages/
│   ├── separate/             # cada um: pyproject.toml próprio; venv em <venvs_dir>/<nome>
│   ├── transcribe/
│   ├── diarize/
│   ├── speakers/
│   ├── translate/
│   ├── synthesize/
│   ├── fit/
│   └── mix/
├── voices/                   # metadados do banco de vozes de fallback (sem áudio no git)
├── tests/
└── docs/
```

Os venvs dos estágios **não** ficam no repositório: ficam em `<venvs_dir>/<nome>` (via `UV_PROJECT_ENVIRONMENT`), no SSD e no mesmo filesystem do cache do `uv`, para aproveitar hardlink/reflink e não duplicar o `torch` ROCm em disco. Só o código fica no `/home`. ([ADR-0014](adr/0014-tudo-no-ssd.md))

## Fora do escopo da v1

- Lip-sync (alterar a boca no vídeo).
- Modo voice-over/ducking (manter o áudio original baixo por baixo da dublagem).
- Processamento em lote, fila ou pasta observada.
- Docker.
- Vídeos acima de 1 hora (podem funcionar, mas não são validados).
