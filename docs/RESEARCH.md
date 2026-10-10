# RESEARCH — Como se constrói um dublador automático de vídeo

> Etapa 1 do plano (`plan.md`): levantamento de como esse tipo de aplicação é construído.
> Data da pesquisa: 2026-10-09. Escopo: pipeline local EN → PT-BR, CLI Linux, vídeos de até 1 h, sem lip-sync na v1.
> Modelos citados aqui são **exemplos**. A escolha de modelos para o hardware fica em `MODELS.md`.
>
> Convenção: **[não verificado]** marca afirmações que não consegui confirmar em fonte primária
> (veio de agregador, blog, memória ou a página não trazia o dado).

---

## Sumário

1. [Pipeline de referência](#1-pipeline-de-referência)
2. [Projetos open-source existentes](#2-projetos-open-source-existentes)
3. [Estado da arte por etapa](#3-estado-da-arte-por-etapa)
4. [Armadilhas conhecidas e mitigações](#4-armadilhas-conhecidas-e-mitigações)
5. [Lições para o dublador](#5-lições-para-o-dublador)
6. [Fontes](#6-fontes)

---

## 1. Pipeline de referência

Quase todos os projetos e papers convergem para a mesma cadeia (a pesquisa da Amazon sobre
*automatic dubbing* descreve exatamente ASR → MT → *prosodic alignment* → TTS):

```
vídeo ─ffmpeg─► áudio ─separação─► vocals ──► ASR (+timestamps) ──► diarização ──► segmentação
                          │                                                         │
                          └──► background (M&E)                     tradução com controle de duração
                                     │                                              │
                                     │                          TTS (clone de voz / banco de vozes)
                                     │                                              │
                                     │                           ajuste de tempo (time-fit)
                                     ▼                                              ▼
                                  mixagem (loudness, ducking) ◄──────────── faixa de voz PT-BR
                                     │
                                  mux (MKV: vídeo copy + PT-BR + EN + SRTs)
```

O que diferencia os projetos não é a ordem das etapas, e sim **como cada um resolve o encaixe de
tempo** (tradução com consciência de duração e aceleração do áudio ou desaceleração do vídeo) e **quanto
deixa o humano editar entre as etapas**.

---

## 2. Projetos open-source existentes

### 2.1 Visão geral

| Projeto | Licença | Interface | Separação | ASR | Diarização | Tradução | TTS / clone | Encaixe de tempo | Saída |
|---|---|---|---|---|---|---|---|---|---|
| **SoniTranslate** (R3gm) | Apache-2.0 (pesos com licenças próprias) | Gradio | UVR/MDX (pasta `mdx_models`) + "vocal enhancement" | WhisperX / faster-whisper | pyannote (desligável, editável) | deep-translator, OpenAI opcional | edge-tts, Piper, XTTS, Bark, MMS; clone via XTTS, OpenVoice V2, FreeVC, RVC | "acceleration rate regulation", "overlap reduction" | MP4/MKV/WAV etc., SRT/ASS, legenda por falante |
| **pyVideoTrans** (jianchang512) | GPL-3.0 | PySide6 GUI, WebUI, **CLI** | "vocal separation" (lib não nomeada) | faster-whisper, WhisperX/Parakeet, Qwen3-ASR (API) etc. | via WhisperX | LLMs (inclui **Ollama**), Google/MS, M2M100 offline | Edge, F5-TTS, CosyVoice, GPT-SoVITS, IndexTTS, Qwen3-TTS, ChatterBox, OmniVoice… | acelera áudio (máx. sugerido 1,8x) **e** desacelera vídeo (até 2x) | vídeo final via ffmpeg |
| **open-dubbing** (Softcatalà) | Apache-2.0 | **CLI** | Demucs | faster-whisper | pyannote | NLLB-200, Apertium | Coqui, MMS, Edge, OpenAI (plugável por API/CLI) | campo `speed` por utterance | vídeo dublado + JSON de metadados |
| **VideoLingo** (Huanshere) | Apache-2.0 | Streamlit + HTTP API | opcional ("pode ajudar") | Qwen3-ASR + Qwen3-ForcedAligner (padrão local), WhisperX | — | LLM: tradução → reflexão → reescrita; glossário de termos | GPT-SoVITS, F5-TTS, CosyVoice2, Edge, Fish… | não detalhado no README | legendas + dublagem |
| **ViDubb** | Apache-2.0 | CLI/notebook | flag `--Bg_sound` (lib não nomeada) | Whisper | pyannote 3.1 | MarianMT; LLM via Groq opcional | clone + emoção (SpeechBrain) | ffmpeg | vídeo, Wav2Lip opcional |
| **Linly-Dubbing** | Apache-2.0 | WebUI | Demucs + UVR5 | WhisperX, FunASR | pyannote 3.1 | OpenAI, Qwen local, Google | Edge, XTTS, CosyVoice, GPT-SoVITS | não descrito | vídeo + legendas, lip-sync via Linly-Talker |
| **YouDub-webui** | (ver repo) | Next.js + FastAPI | Demucs | openai-whisper (large-v3-turbo) | — | API compatível OpenAI (pré-processa o texto inteiro, depois traduz em paralelo) | VoxCPM2 | `audiostretchy`, librosa, pydub | MP4 |
| **Mazinger** (bakrianoo) | MIT | CLI + Gradio | Demucs (stem de voz também alimenta VAD) | faster-whisper, CohereX, Deepgram | via pyannote (CohereX) | LLM via **Ollama** ou nuvem | Qwen3-TTS, OmniVoice, Chatterbox; clone com referência de 20–60 s | **"speech map"** via VAD: cada fala começa onde a original começou e cabe no span de fala original | MP4, legendas queimadas |
| **OmniVoice Studio** | AGPL-3.0 | desktop | Demucs **[não verificado]** | Whisper **[não verificado]** | pyannote/WhisperX **[não verificado]** | — | OmniVoice (k2-fsa) | "timed speech" | — |

Observações:
- Números de estrelas e atividade mudam rápido; não os uso como critério. O OmniVoice Studio
  aparecia com contagem de estrelas atipicamente alta na captura; a descrição do pipeline dele vem
  de blog de terceiros **[não verificado]**.
- O **google/ariel** (projeto do Google que inspirou parte do open-dubbing) retornou 404 no
  GitHub; provavelmente foi removido ou renomeado **[não verificado]**.

### 2.2 Detalhes e o que dá para aproveitar

**SoniTranslate.** É o mais completo em opções de TTS e clone. Vale copiar três ideias: (a) separação
de voz *antes* do ASR; (b) **edição manual de falantes** depois da diarização; (c) controles
explícitos de "taxa máxima de aceleração" e "redução de overlap" entre segmentos vizinhos. A última
notícia do README é de maio de 2024, então parece pouco ativo. O próprio README avisa que os pesos
(pyannote, por exemplo) podem ter restrições comerciais.

**pyVideoTrans.** É o projeto mais ativo e com mais motores. Tem **CLI headless** e **pausa para
revisão humana** depois do ASR, da tradução e da dublagem. A documentação de sincronização admite que
"nenhum método automático alcança sincronia perfeita" e recomenda combinar aceleração do áudio (até
1,8x) com desaceleração do vídeo (até 2x) e um *speed-up* global de ~10%. Para nós, o útil é ver
**quanto** de aceleração os usuários toleram na prática. O 1,8x deles é bem mais agressivo que o
nosso 1,25x, e eles têm a alavanca extra de desacelerar o vídeo, que nós descartamos ao fazer
*stream copy*.

**open-dubbing.** É o mais próximo do nosso escopo: CLI, Demucs, faster-whisper, pyannote e
**detecção de gênero para escolher a voz sintética**. O melhor padrão a copiar é o **JSON de
metadados por utterance** (`start`, `end`, `speaker_id`, `text`, `translated_text`, `gender`,
`assigned_voice`, `speed`, `path`, `dubbed_path`, `hash`) com a flag `--update`, que **reconstrói só o
que mudou** depois de uma edição manual. Ele também documenta limitações: os erros se acumulam etapa
a etapa, e a detecção de idioma falha se os primeiros 30 s forem só música. O README não diz qual
modelo detecta o gênero **[não verificado]**.

**VideoLingo.** O foco é a **qualidade da tradução e da legenda**: segmentação por NLP+LLM com
limite de comprimento, glossário de termos (custom + gerado por IA) para consistência, e tradução
em etapas (traduzir → refletir → reescrever natural). Também tem **retomada de progresso**. Para nós
servem o glossário por vídeo e a tradução em duas passadas (fiel, depois adaptada).

**YouDub-webui.** Faz um **pré-processamento do texto inteiro** (resumo/contexto) antes de traduzir as
frases em paralelo. É o mesmo princípio do nosso "contexto de diálogo" para o LLM. Para ajustar o
tempo usa `audiostretchy` (TDHS), uma alternativa leve ao rubberband.

**Mazinger.** É a ideia de encaixe mais interessante: um **speech map** construído com VAD sobre o stem
de voz do Demucs. Cada fala dublada **começa onde a fala original começou** e precisa caber no span
de fala real, não no span da legenda. Também preserva sons curtos não transcritos (risadas) do
original, tem **etapas independentes e cacheadas** (até cada segmento de TTS) e um editor por
*chunk* que marca o que ficou desatualizado sem re-executar tudo sozinho. Usa Ollama para traduzir.

**ViDubb / Linly-Dubbing.** São úteis como referência de "pipeline com emoção" (SpeechBrain) e
lip-sync (Wav2Lip), coisas fora do nosso escopo na v1. Linly usa Demucs **e** UVR5, sinal de que
Demucs sozinho não bastava para eles.

### 2.3 Padrões recorrentes entre os projetos

| Padrão | Quem faz | Relevância para nós |
|---|---|---|
| Separação antes do ASR | quase todos | confirma a nossa decisão |
| Metadados editáveis + rebuild incremental | open-dubbing, Mazinger, pyVideoTrans | **alta**: vídeos de 1 h falham no meio |
| Cache por etapa/segmento, retomada | Mazinger, VideoLingo | **alta** |
| Tradução com contexto global (resumo, glossário) | YouDub, VideoLingo | alta |
| Encaixe por aceleração com teto | SoniTranslate, pyVideoTrans | confirma; tetos variam de 1,25x a 1,8x+ |
| Desacelerar vídeo | pyVideoTrans | **incompatível** com o nosso *stream copy* |
| Âncora no início real da fala (VAD) | Mazinger | alta: melhor que o timestamp do segmento ASR |
| Revisão humana opcional entre etapas | pyVideoTrans, SoniTranslate | média: modo `--review` futuro |

---

## 3. Estado da arte por etapa

### 3.1 Extração e separação de fontes

| Técnica | Notas |
|---|---|
| **Demucs v4 (htdemucs / htdemucs_ft)** | Padrão de fato nos projetos de dublagem. O repositório `facebookresearch/demucs` foi **arquivado em 01/01/2025**. O autor mantém um fork em `adefossez/demucs` que recebe só correções importantes. Um guia de terceiros cita uma versão 4.1.0 do fork em 2026 **[não verificado]**. |
| **BS-RoFormer / Mel-RoFormer** (via `python-audio-separator` / UVR) | Comunidade UVR. Os checkpoints trazem SDR de vocais no nome (ex.: BS-RoFormer viperx ~12,9). São SDRs **auto-reportados**, sem benchmark comum com o htdemucs aqui **[não verificado]**. Na prática da comunidade, RoFormers são considerados superiores para vocais. |
| **MDX-Net / UVR-MDX** | Usado por SoniTranslate; também há modelos de *de-reverb* e *de-echo* para limpar o stem de voz. |
| Cadeia em dois estágios | Separar vocais e depois aplicar de-reverb/denoise **só na cópia usada para ASR/referência de clone**, mantendo o stem original para outras finalidades. |

Pontos práticos:
- Processar 1 h de áudio em **chunks com overlap** (o Demucs já faz isso com `--segment`/`overlap`) para limitar VRAM.
- O stem "no_vocals" quase sempre tem **vazamento de voz** (fantasma do inglês), principalmente com reverb. Ver §4.

### 3.2 ASR com timestamps por palavra

| Abordagem | Notas |
|---|---|
| Whisper (faster-whisper / CTranslate2) | Timestamps nativos por segmento são imprecisos ("podem errar por vários segundos", README do WhisperX). `word_timestamps=True` usa atenção cruzada (DTW), que funciona razoavelmente. |
| **WhisperX** | Whisper + **alinhamento forçado com wav2vec2** por idioma, o que dá timestamps por palavra mais precisos. Limitações documentadas: tokens fora do dicionário do alinhador (números, "£13.60") ficam sem tempo, e **fala sobreposta não é bem tratada**. |
| Qwen3-ASR + Qwen3-ForcedAligner | Alternativa atual usada pelo VideoLingo como padrão local. |
| Parakeet / Canary (NVIDIA) | Alternativas rápidas para inglês, citadas pelo pyVideoTrans. |

Técnicas comuns:
- **VAD antes do ASR** (Silero) para cortar silêncio e reduzir alucinação (ver §4.4).
- Segmentar as frases para tradução **por pontuação e pausas** (NLTK/spaCy ou LLM, como no VideoLingo), não pelos segmentos de 30 s do Whisper.
- Guardar as palavras com tempo: são a base para atribuir falante por palavra e para âncoras de VAD.

### 3.3 Diarização e clustering de falantes

| Técnica | Notas |
|---|---|
| **pyannote `speaker-diarization-community-1`** (set/2025) | Sucessor do 3.1, com melhor atribuição e contagem de falantes. Tem saída **`exclusive_speaker_diarization`** (um falante por vez), que simplifica casar turnos com timestamps do ASR. Licença CC-BY-4.0, exige aceitar termos no HF + token, roda offline depois de baixado. DER menor que o 3.1 na maioria dos benchmarks (ex.: AMI-IHM 18,8 → 17,0; AVA-AVD 49,7 → 44,6). |
| pyannote 3.1 | Ainda é o mais usado (ViDubb, Linly, WhisperX). |
| NVIDIA Streaming Sortformer → **Nemotron 3 Diarization** (23/09/2026) | Trata overlap nativamente, até 8 falantes, licença OpenMDW. O DIHARD III da model card é número do próprio fornecedor **[não verificado independentemente]**. |
| Atribuição palavra→falante | WhisperX `assign_word_speakers`: maior sobreposição temporal entre palavra e turno. Re-segmentar quando o falante muda no meio de uma frase do ASR. |
| Pós-processamento | Juntar clusters muito parecidos (similaridade de cosseno entre embeddings médios, ex.: ECAPA/WeSpeaker), descartar "falantes" com poucos segundos ou reatribuí-los ao vizinho mais próximo, e permitir `num_speakers`/`min/max_speakers` via CLI. |

Ponto de atenção: diarizar sobre o **stem de voz separado** em vez do mix original. Isso tira música
e efeitos (bom), mas artefatos de separação podem mexer nos embeddings. Nenhum projeto documenta
comparação. Vale testar os dois **[não verificado]**.

### 3.4 Detecção de gênero do falante

| Técnica | Notas |
|---|---|
| Classificador neural sobre wav2vec2 (ex.: `audeering/wav2vec2-large-robust-*-ft-age-gender`) | Prediz child/female/male + idade. **Licença CC-BY-NC-SA 4.0 (não comercial)**. |
| Heurística por F0 (pitch mediano) | Barata e sem modelo: grosso modo, masculino tem F0 de ~85–155 Hz e feminino de ~165–255 Hz **[não verificado: valores de referência comuns da literatura de fonética, não conferidos aqui]**. Falha com vozes graves femininas, crianças e falas gritadas. |
| Classificador sobre embedding de falante | Treinar um regressor logístico simples sobre embeddings ECAPA/WeSpeaker que a diarização já calcula. Custo marginal quase zero. |
| open-dubbing | Faz detecção de gênero para escolher a voz, mas o README não diz como **[não verificado]**. |

Boa prática: decidir o gênero **por cluster de falante** (agregando vários segmentos e votando),
nunca por segmento. Guardar a confiança e permitir override manual no JSON de metadados.

### 3.5 Tradução com LLM e controle de comprimento (isocronia)

O problema central do dublador é que PT-BR é mais longo que EN (§4.3) e **LLMs tendem a ser
"tagarelas"**: o paper HOMURA observa que as traduções de LLM saem significativamente mais longas que
a fonte mesmo sem mudança de sentido.

| Linha de pesquisa | Ideia | Achado relevante |
|---|---|---|
| **VideoDubber** (Wu et al., 2022) | MT com controle de comprimento ciente da **duração de fala por token** | Contar caracteres ou palavras é insuficiente, porque a duração por caractere varia entre línguas. |
| Amazon: *Prosodic Alignment* (Virkar et al., 2021–22), isocronia com pause markers (Tam et al., 2022), otimização conjunta tradução+timing (arXiv 2302.12979) | Segmentar a tradução para seguir o padrão frase/pausa da fonte; relaxar restrições para falante **off-screen** | Para conteúdo sem lip-sync (o nosso caso), a restrição pode ser mais frouxa. |
| **IsoChronoMeter** (arXiv 2410.11127) | Métrica de isocronia usando preditores de duração de TTS | Traduções "normais", mesmo de LLMs SOTA, **não alcançam boa isocronia**. |
| **Javorský, Bojar, Yvon (2025)** — prompting para MT isométrica | 8 LLMs abertos sob prompts de comprimento | LLMs só encurtam com exemplos extremos; demonstrações isométricas são ignoradas; **gerar múltiplas saídas e escolher** melhora muito o trade-off. |
| **Duration-based Translation** (Won et al., EMNLP 2025 demo) | Prever **nº ótimo de fonemas** pela duração da fonte e **ajustar iterativamente** a tradução | +24% relativo de overlap de fala, COMET competitivo. |
| **SSPO** (ACL 2025) | Preference optimization por segmento | Prompt e SFT sozinhos resolvem mal, porque o LLM "não sabe" a duração falada do texto. |
| **HOMURA / Sand-Glass** (arXiv 2601.10187) | RL com recompensa de razão de sílabas | Prompt engineering tem dificuldade com restrição rígida de tempo. |
| Descript (estudo de caso OpenAI) | Estimar o **nº de sílabas alvo** com taxa de fala por idioma e pedir ao LLM duração + sentido | **Contar sílabas de forma confiável foi crítico** (página retornou 403; resumo via busca) **[não verificado na fonte]**. |

Receita prática que emerge (sem fine-tuning):
1. Calcular para cada segmento o **orçamento** = duração disponível (span da fala original + folga até o próximo segmento) × taxa de fala PT-BR (sílabas/s), e passá-lo ao LLM como **nº de sílabas alvo** (não caracteres).
2. Dar **contexto**: resumo do vídeo, glossário, segmentos anteriores e posteriores, falante e gênero (concordância de gênero em PT!).
3. **Gerar N candidatos** (ou pedir versões "normal" e "curta"), contar sílabas **com código** (o LLM conta mal) e escolher o que cabe com melhor fidelidade.
4. Se ainda não couber: pedir reescrita mais curta explicitando o excesso ("reduza ~4 sílabas"), em loop limitado.
5. Opcional: medir a duração **real** após o TTS e voltar ao passo 4 se passar do teto de stretch.

### 3.6 TTS, clone de voz e seleção de áudio de referência

- Zero-shot TTS atual (exemplos: F5-TTS, CosyVoice, XTTS-v2, IndexTTS2, Qwen3-TTS, OmniVoice, VoxCPM, Chatterbox) clona a voz com **3–15 s de referência**.
- **Alguns modelos aceitam duração alvo** (IndexTTS2 tem modo explícito com nº de tokens e erro < 0,03% entre 0,875x e 1,125x, e WER praticamente estável até 0,75x/1,25x; F5-TTS, por ser não-autoregressivo, também gera para uma duração dada). Isso permite **gerar já no tamanho certo** em vez de esticar depois, com prosódia mais natural que o time-stretch.

Seleção de referência (convergência de docs de F5-TTS, CosyVoice/Fish e guias de clone):

| Critério | Recomendação |
|---|---|
| Duração | ~6–12 s (F5-TTS corta em ~12–15 s; referência + saída ≤ ~30 s por geração) |
| Pureza | um único falante, **sem overlap**, sem música (usar stem de voz) e sem reverb forte |
| Fim do clipe | terminar em fronteira de palavra, com ~1 s de silêncio (F5-TTS) |
| Transcrição | fornecer o **texto exato** da referência (já temos do ASR) em vez de re-transcrever |
| Seleção automática | ranquear segmentos do falante por: duração adequada, distância ao centróide do embedding do falante (pureza), SNR/energia, ausência de overlap na diarização, confiança do ASR; concatenar 2–3 trechos curtos se nenhum for longo o bastante |
| Estilo | evitar sussurro e grito; uma referência "neutra" por falante é mais estável (a mesma voz para o vídeo inteiro) |
| Cross-lingual | a referência é EN e a saída PT-BR, então o sotaque inglês pode "vazar" para o PT em alguns modelos (ver §4) |

Fallback: banco de vozes por gênero, como fazem open-dubbing e SoniTranslate com Edge/Piper. Atribuir
vozes **distintas** a falantes distintos do mesmo gênero (round-robin no banco), de forma estável ao
longo do vídeo.

### 3.7 Alinhamento de duração / time-stretching

| Ferramenta | Notas |
|---|---|
| ffmpeg `atempo` | Leve (WSOLA), 0,5–2,0 por instância (encadear para mais). Qualidade aceitável até ~1,2–1,3x em fala. |
| ffmpeg `rubberband` / `rubberband` CLI | Motor **R3** ("finer") com qualidade melhor e mais CPU. Opções de preservação de formantes (relevante para pitch, não para tempo puro). Um relato antigo (2018) reclama de artefatos com o motor R2. |
| `audiostretchy` (TDHS) | Usado no YouDub; leve, pensado para fala. |
| Controle de duração no próprio TTS | Ver §3.6. Preferível ao stretch quando disponível. |
| Desacelerar vídeo | pyVideoTrans; **não se aplica** a nós (stream copy). |

Algoritmo de encaixe (síntese de SoniTranslate, pyVideoTrans, Mazinger e Amazon PA):
1. **Âncora** = início real da fala original (VAD sobre o stem de voz), não o início do segmento ASR.
2. **Janela disponível** = do início até o próximo segmento (de qualquer falante) menos uma margem mínima (~100–200 ms).
3. Se o TTS couber: posicionar e, se sobrar, deixar silêncio (não esticar para "preencher").
4. Se não couber: (a) re-traduzir mais curto → (b) re-gerar com duração alvo/taxa do TTS → (c) stretch ≤ teto → (d) **invadir o silêncio seguinte** → (e) último recurso: deslocar o início alguns ms para trás se houver silêncio antes, ou aceitar sobreposição com *fade*/truncamento sinalizado no log.
5. Registrar no JSON o ratio usado e o que "não coube", para revisão.

### 3.8 Mixagem

| Técnica | Notas |
|---|---|
| Normalização de loudness | EBU R128: −23 LUFS integrado (broadcast). Netflix: **−27 LKFS dialog-gated ±2 LU**, true peak ≤ −2 dBTP. Para arquivo pessoal/streaming, algo entre −16 e −23 LUFS é comum **[não verificado como norma]**. ffmpeg `loudnorm` (duas passadas para precisão). |
| Casar nível da voz | Normalizar **cada clipe TTS** para o loudness da fala original correspondente (ou do falante), preservando a dinâmica relativa (gritos/sussurros). |
| Ducking | Abaixar o background **só enquanto há voz dublada** (ffmpeg `sidechaincompress`, ou envelope de ganho calculado a partir dos tempos dos segmentos). Mascara o resíduo de voz inglesa no stem de fundo. |
| Atenuação de resíduo | Nos spans onde havia fala original, aplicar redução extra (−3 a −6 dB) no background **[valor heurístico, não verificado]**. |
| Preservar não-verbais | Risadas, suspiros e gritos curtos não transcritos: manter do stem de voz original (Mazinger faz isso). |
| Crossfades | Fades curtos (5–20 ms) em cada clipe para evitar cliques. |
| Limiter final | True-peak limiter (≈ −1 a −2 dBTP). |

### 3.9 Legendas

- **EN SRT**: direto do ASR com timestamps por palavra, re-segmentado por regras de legibilidade.
- **PT-BR SRT**: do texto traduzido, com tempos do **áudio dublado final** (depois do time-fit), não do original, senão a legenda descola da voz.
- Regras Netflix PT-BR: **máx. 42 caracteres/linha**, 2 linhas. Velocidade de leitura de 17 CPS (adulto) vem do guia "Portuguese" e não foi confirmada no guia pt-BR **[não verificado]**.
- A legenda pode ser **menos condensada** que a fala dublada: VideoLingo e pyVideoTrans tratam "texto para legenda" e "texto para TTS" como variações do mesmo segmento. Gerar as duas é barato.
- MKV: SRT como trilhas de legenda com `language=por`/`eng` e títulos; dispensa queimar.

---

## 4. Armadilhas conhecidas e mitigações

### 4.1 Fala sobreposta (overlap)
- **Problema:** Whisper/WhisperX transcrevem mal; a diarização atribui errado; a referência de clone fica contaminada; o TTS gera falas que se atropelam.
- **Mitigações:** usar a saída *exclusive* da diarização (community-1) para atribuir palavras; **excluir trechos de overlap das referências de clone**; no encaixe, permitir sobreposição dublada só quando a original também sobrepunha; marcar segmentos com overlap no JSON para revisão. Modelos que tratam overlap (Sortformer/Nemotron) são uma opção futura.

### 4.2 Falantes curtos / com pouco áudio limpo
- **Problema:** um falante com 1–2 falas curtas não tem referência ≥ 3–6 s; a diarização cria "falantes fantasmas" a partir de ruído ou risadas.
- **Mitigações:** limiar mínimo de áudio limpo para clonar (ex.: ≥ 6 s somados) e, abaixo dele, o **banco de vozes por gênero** (já decidido); fundir clusters pequenos com o vizinho mais similar; concatenar vários trechos curtos se a soma bastar e forem puros.

### 4.3 PT-BR é mais longo que EN
- **Dado:** agências de localização estimam **+15% a +30%** de expansão EN→PT em texto (algumas dizem 10–15%). A taxa de fala em sílabas/s também varia entre línguas (Pellegrino et al., 2011), então a expansão **em tempo de fala** não é igual à expansão em caracteres **[não verificado: não há número EN→PT-BR em tempo]**.
- **Consequência:** um teto de stretch de 1,25x cobre cerca do limite superior da expansão típica, **sem margem** para frases já rápidas no original. A tradução encurtada precisa fazer a maior parte do trabalho.
- **Mitigações:** orçamento de sílabas no prompt, N candidatos, loop de reescrita, TTS com duração alvo, invadir silêncio seguinte (ver §3.5, §3.7).

### 4.4 Alucinações do Whisper em silêncio/música
- **Problema:** frases fantasma ("Thank you for watching!", "Legendas por…"), loops repetitivos em silêncio, ruído ou música.
- **Mitigações:** `vad_filter=True` (Silero) e `condition_on_previous_text=False` (evita propagar loops, ao custo de menos coerência entre segmentos); `hallucination_silence_threshold` com `word_timestamps=True`; filtros pós-ASR (repetição/n-gram, lista de frases-fantasma conhecidas, segmentos com `no_speech_prob` alto ou `avg_logprob` baixo); rodar o ASR sobre o **stem de voz** (menos música). Uma issue do faster-whisper relata que **denoise piorou** a alucinação em quase-silêncio, então convém testar e não aplicar às cegas.

### 4.5 Música/efeitos vazando para o stem de voz (e voz vazando para o fundo)
- **Problema:** música no stem de voz contamina ASR, diarização e **referência de clone** (o TTS reproduz o "ruído"); voz residual no fundo vira um "fantasma" do inglês sob o PT.
- **Mitigações:** modelo de separação mais forte (RoFormer) ou ensemble; de-reverb só para referência; selecionar referências com alta pureza; **ducking** e atenuação extra do fundo nos spans de fala.

### 4.6 Outros
| Armadilha | Mitigação |
|---|---|
| Detecção de idioma falha com intro musical (open-dubbing) | Forçar `language="en"` (sabemos a origem). |
| Números, siglas e nomes próprios | Normalizar o texto para o TTS ("2024" → "dois mil e vinte e quatro") e manter glossário de nomes que **não** devem ser traduzidos. |
| Concordância de gênero em PT ("obrigado/obrigada") | Passar gênero do falante **e** do interlocutor ao LLM. |
| Sotaque inglês "vazando" no clone cross-lingual | Testar por modelo; se forte, usar referência mais curta ou a voz do banco. |
| Instabilidade de TTS (palavras puladas/repetidas, cauda de ruído) | Validar cada clipe re-transcrevendo com ASR (WER/CER vs. texto), e regenerar com outra seed se falhar. Aparar silêncios nas bordas. |
| Erros em cascata | Metadados editáveis + rebuild incremental (open-dubbing `--update`). |
| Vídeos de 1 h: memória e falhas | Chunking na separação e no ASR; cache por etapa e por segmento; retomada. |
| Licenças de pesos | pyannote (termos HF), audeering (NC), Coqui XTTS (CPML, não comercial) **[não verificado no momento]**: registrar licença de cada peso. |
| Consentimento / clone de voz | Uso pessoal; documentar no README. |

---

## 5. Lições para o dublador

Mapeamento dos achados para a arquitetura decidida. ✔ = confirma, ⚠ = ajuste recomendado, ✖ = contradiz.

| Etapa decidida | Veredito | Lição |
|---|---|---|
| ffmpeg → separação Demucs-class | ✔ / ⚠ | Padrão em quase todos os projetos. **Demucs está arquivado** (fork mantido só com correções). Deixar a separação atrás de uma interface e avaliar RoFormer via `audio-separator`. |
| ASR + diarização nos vocals limpos | ✔ / ⚠ | Confirmado (SoniTranslate, Linly, open-dubbing). Usar timestamps por palavra (alinhamento forçado), VAD + `condition_on_previous_text=False`, forçar `en`. Diarização: preferir saída *exclusive*; testar diarizar no stem e no mix. |
| Gênero por falante | ✔ / ⚠ | Decidir **por cluster**, com confiança e override. Atenção à licença NC do classificador audeering; F0 ou classificador sobre embedding são alternativas livres. |
| Tradução via Ollama com contexto + duração alvo | ✔ / ⚠ | Direção certa, mas a literatura é unânime: **prompt sozinho controla mal o comprimento**. Precisamos de: orçamento em **sílabas** contado por código, **N candidatos + seleção**, loop de reescrita e glossário/resumo global. |
| TTS zero-shot por falante + fallback banco por gênero | ✔ | Confirmado (open-dubbing, SoniTranslate). Adicionar seleção automática de referência (pureza, 6–12 s, sem overlap, texto exato) e validação do clipe por ASR. Vozes distintas por falante no fallback. |
| Time-fit: encurtar → stretch ≤ 1,25x → silêncio | ⚠ | Ordem coerente com a literatura. Três ajustes: (1) inserir **"gerar com duração alvo no TTS"** antes do stretch quando o modelo suportar (IndexTTS2/F5-class), porque soa melhor que esticar; (2) ancorar no **início real da fala via VAD** (Mazinger); (3) 1,25x é conservador frente ao 1,8x do pyVideoTrans. Bom para qualidade, mas com +15–30% de expansão o encurtamento vira o gargalo, então o teto deve ser configurável. |
| Remix com background preservado | ⚠ | Faltam na decisão: **ducking**/atenuação do fundo nos spans de fala (resíduo de inglês), loudness por clipe casado com o original, preservar não-verbais, `loudnorm` + limiter. |
| Mux MKV (PT-BR + EN, SRT EN/PT-BR, vídeo copy) | ✔ | Sem conflito. **Copiar o vídeo elimina a alavanca "desacelerar vídeo"** do pyVideoTrans: decisão consciente, que reforça a dependência da tradução curta. Legenda PT-BR com tempos do áudio **dublado**; regra de 42 caracteres/linha. |
| Sem lip-sync na v1 | ✔ | A literatura de *off-screen dubbing* (Amazon) mostra que sem lip-sync a isocronia pode ser relaxada: permitir pequenas invasões de silêncio é legítimo. |
| (não decidido) Infra do pipeline | ⚠ **adicionar** | Todos os projetos maduros têm: **JSON de metadados por segmento**, **cache/retomada por etapa e por segmento** e **rebuild incremental após edição manual**. Para vídeos de 1 h isso é praticamente obrigatório. |

**Nada encontrado contradiz frontalmente a arquitetura.** Os pontos de maior risco são (a) controle
de comprimento só por prompt, (b) teto de 1,25x sem a alavanca de vídeo e (c) resíduo de voz no
background sem ducking.

---

## 6. Fontes

### Projetos
- SoniTranslate — https://github.com/R3gm/SoniTranslate
- pyVideoTrans — https://github.com/jianchang512/pyvideotrans
- pyVideoTrans, doc de alinhamento — https://en.pyvideotrans.com/align
- open-dubbing — https://github.com/Softcatala/open-dubbing
- VideoLingo — https://github.com/Huanshere/VideoLingo
- ViDubb — https://github.com/medahmedkrichen/ViDubb
- Linly-Dubbing — https://github.com/Kedreamix/Linly-Dubbing
- YouDub-webui — https://github.com/liuzhao1225/YouDub-webui
- Mazinger — https://github.com/bakrianoo/mazinger
- OmniVoice Studio — https://github.com/debpalash/OmniVoice-Studio
- OmniVoice Studio (blog de terceiros) — https://themenonlab.blog/blog/omnivoice-studio-open-source-local-dubbing-646-languages
- google/ariel (404 em 2026-10-09) — https://github.com/google/ariel

### Separação
- Demucs (arquivado) — https://github.com/facebookresearch/demucs
- Demucs fork mantido — https://github.com/adefossez/demucs
- python-audio-separator / modelos RoFormer (lista em Space HF) — https://huggingface.co/spaces/next-playground/UVR-API/blame/b0592e697a63223d107f966440c20bf7b72d787f/app.py

### ASR / diarização / gênero
- WhisperX — https://github.com/m-bain/whisperx
- faster-whisper, issue sobre VAD e quase-silêncio — https://github.com/SYSTRAN/faster-whisper/issues/843
- Discussão de alucinação no Whisper — https://github.com/openai/whisper/discussions/679
- pyannote community-1 — https://huggingface.co/pyannote/speaker-diarization-community-1
- NVIDIA Streaming Sortformer — https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- Sortformer 4spk v2.1 — https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- Nemotron 3 Diarization (cobertura) — https://www.unite.ai/nvidia-releases-nemotron-3-diarization-open-weight-speaker-model/
- audeering age/gender — https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender

### Tradução isócrona
- VideoDubber (Wu et al.) — https://awesomepapers.io/speech-audio/papers/wu2022videodubber
- Duration-based Translation (Won et al., EMNLP 2025 demos) — https://aclanthology.org/2025.emnlp-demos.37
- SSPO (ACL 2025) — https://preview.aclanthology.org/landing_page/2025.acl-long.227.pdf
- Javorský et al., length control prompting — https://arxiv.org/pdf/2506.04855v1
- HOMURA / Sand-Glass — https://arxiv.org/html/2601.10187v1
- IsoChronoMeter — https://arxiv.org/html/2410.11127v1
- Jointly Optimizing Translations and Speech Timing — https://arxiv.org/pdf/2302.12979
- Prosodic Alignment for off-screen dubbing — https://arxiv.org/pdf/2204.02530
- Isochrony-aware MT (Tam et al., Interspeech 2022) — https://www.isca-archive.org/interspeech_2022/tam22_interspeech.pdf
- Duration modeling of neural TTS for automatic dubbing — https://assets.amazon.science/52/07/585fa2364bb39d8661326ab19bc7/duration-modeling-of-neural-tts-for-automatic-dubbing.pdf
- Descript (estudo de caso OpenAI; 403 no fetch) — https://openai.com/index/descript/

### TTS / clone
- IndexTTS2 (controle de duração) — https://arxiv.org/abs/2506.21619
- F5-TTS, README de inferência — https://huggingface.co/spaces/mrfakename/E2-F5-TTS/raw/630814d220ec334918462236ca35f5d96f09ceee/src/f5_tts/infer/README.md
- VoxKitchen, guia de clone (CosyVoice/Fish) — https://voxkitchen.readthedocs.io/en/latest/tutorials/tts-voice-cloning/

### Time-stretch / mixagem / legendas
- Rubber Band, uso — https://breakfastquay.com/rubberband/usage.txt
- Rubber Band man page — https://dyn.manpages.debian.org/trixie/rubberband-cli/rubberband.1
- ffmpeg-devel, limites do atempo — https://ffmpeg.org/pipermail/ffmpeg-devel/2018-June/230964.html
- EBU R 128 — https://tech.ebu.ch/fr/loudness
- Netflix Sound Mix Specifications — https://backlothelp.netflix.com/hc/en-us/articles/360001794307-Netflix-Sound-Mix-Specifications-Best-Practices-v1-5
- Netflix PT-BR Timed Text Style Guide — https://backlothelp.netflix.com/hc/en-us/articles/215600497-Portuguese-Brazil-Timed-Text-Style-Guide
- Netflix Portuguese Timed Text Style Guide — https://backlothelp.netflix.com/hc/en-us/articles/216787938-Portuguese-Timed-Text-Style-Guide

### Expansão de texto / taxa de fala
- Andiamo, fatores de expansão — https://www.andiamo.co.uk/resources/expansion-and-contraction-factors/
- Argotrans, expansão — https://argotrans.com/blog/text-expansion-during-translation
- Pellegrino, Coupé & Marsico (2011), *Language* — https://www.cambridge.org/core/journals/language/article/abs/crosslanguage-perspective-on-speech-information-rate/73064B3F77E7CE81C42CAB29B4393F80
