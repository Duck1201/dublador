# MODELS — Escolha de modelos locais do `dublador`

> Levantamento feito em **2026-10-09** para a máquina-alvo (Ryzen 5 5600, 32 GB RAM, **RX 7600 8 GB / gfx1102**, Debian 13, kernel `6.12.111+deb13-rt-amd64`).
> Pipeline: EN → PT-BR, 100% local, um venv `uv` por estágio, **um modelo na VRAM por vez**, uso pessoal (licenças não comerciais aceitas, mas registradas).
>
> Legenda: **[V]** = verificado nesta pesquisa (fonte primária ou checagem na máquina) · **[NV]** = não verificado / estimativa / relato de terceiros — **testar** (ver [Tarefas de validação](#tarefas-de-validação)).

---

## Resumo executivo (defaults recomendados)

| # | Estágio | Default recomendado | Alternativas | Licença do default | Bloqueio / risco principal |
|---|---------|--------------------|--------------|-------------------|----------------------------|
| 1 | Separação voz × fundo | **BS-RoFormer-Viperx-1297** (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) via `audio-separator` | Mel-RoFormer Kim (vocals), `htdemucs_ft`, MDX23C-InstVoc HQ | MIT (código); pesos comunitários UVR, licença informal [NV] | Modelos treinados em música — SFX/ruído de filme pode vazar pro stem de voz [NV] |
| 2 | ASR EN + timestamps por palavra | **WhisperX** (faster-whisper `large-v3` + alinhamento wav2vec2) com **CTranslate2 ROCm** | Whisper large-v3 via `transformers` (PyTorch ROCm); Qwen3-ASR-1.7B + Qwen3-ForcedAligner; whisper.cpp Vulkan | MIT / BSD-2 / Apache-2.0 | Wheel ROCm do CTranslate2 exige **libs ROCm 7.2 no sistema** (não embutidas); fallback CPU int8 |
| 3 | Diarização | **pyannote `speaker-diarization-community-1`** (pyannote.audio 4.0.x) | pyannote 3.1; DiariZen; NVIDIA Sortformer | CC-BY-4.0 (gated, token HF só no 1º download) | `torchcodec` precisa casar com a versão do torch |
| 4 | Gênero do falante | **`JaesungHuh/voice-gender-classifier`** (ECAPA-TDNN) + voto por falante | `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`; `audeering/wav2vec2-large-robust-24-ft-age-gender` (detecta criança); inaSpeechSegmenter | MIT | Só binário M/F; crianças/vozes ambíguas → regra de fallback |
| 5 | Tradução (Ollama) | **`gemma4:e4b-it-qat`** (provisório) / qualidade: **`qwen3:30b-a3b-instruct-2507-q4_K_M`** | `gemma4:26b-a4b-it-qat`, `translategemma:12b`/`4b` | Gemma ToU / Apache-2.0 | ⚠ O `qwen3:30b-a3b` instalado é o **Thinking-2507** (não dá pra desligar o raciocínio) — puxar o *instruct-2507* |
| 6 | TTS com clonagem PT-BR | **OmniVoice** (`k2-fsa/OmniVoice`), clonando com referência curta de 3–4,5 s ([ADR-0017](adr/0017-tts-omnivoice-e-referencia-curta.md)) | Chatterbox pt-br e Qwen3-TTS (saíram: robóticos e com sotaque), dots.tts MF e F5-TTS pt-br (perderam no bake-off), **FireRedTTS3**, **MOSS-TTS-Local**, **VoxCPM2** | código Apache-2.0; **pesos CC-BY-NC** | Clonar de referência em inglês transfere sotaque; com 3,4 s o sotaque fica aceitável ("MID") |
| 7 | Banco de vozes fallback | Vozes geradas por **Qwen3-TTS VoiceDesign** + referências de **CML-TTS/MLS pt** (CC-BY-4.0) | Kokoro (`pf_dora`, `pm_alex`, `pm_santa`); Common Voice pt (CC0) | Apache-2.0 / CC-BY-4.0 | Curadoria manual de sotaque BR |
| 8 | Time-stretch | **Rubber Band** (`rubberband` CLI / `pyrubberband`, modo `--fine`/R3) | `ffmpeg atempo` (fallback), `librosa` | GPL-2.0 (CLI chamado como subprocesso) | Limitar a ~0,80–1,25× para não soar artificial |

**Riscos bloqueantes (resumo):**
1. **Espaço em `/home`:** só ~31 GB livres e o cache do `uv` já ocupa 18 GB em `~/.cache/uv`. Um único wheel `torch+rocm7.2` tem **6,2 GB compactado** [V]. Mover `UV_CACHE_DIR`, venvs e `HF_HOME` para o SSD (`/home/duck/SSD/Extras`, btrfs, 230 GB livres) **antes** de instalar qualquer coisa. → Decidido na [ADR-0014](adr/0014-tudo-no-ssd.md).
2. **Ollama segura a VRAM:** o serviço está com `OLLAMA_KEEP_ALIVE=5m` e `OLLAMA_MAX_LOADED_MODELS=3` [V]. Sem `keep_alive: 0` no último request (ou `ollama stop <modelo>`), o LLM fica na VRAM e o estágio de TTS dá OOM.
3. **Qwen3 instalado é thinking-only** [V] (digest `ad815644918f` = `qwen3:30b-a3b-thinking-2507-q4_K_M`). `think:false` não resolve; usar `qwen3:30b-a3b-instruct-2507-q4_K_M` → decidido na [ADR-0015](adr/0015-tradutor-qualidade-instruct.md).
4. **`torchaudio` parou na 2.11** [V] (PyPI e índices ROCm). Pacotes que exigem `torchaudio` devem usar `torch==2.11.0+rocm7.2` (não 2.12+). Vários pacotes também têm pins rígidos (Chatterbox `torch==2.6.0`, WhisperX `torch~=2.8.0`) → usar `override-dependencies` do uv ou a versão ROCm correspondente.
5. **gfx1102 não está na lista oficial de suporte Radeon do ROCm** [V], mas os wheels oficiais do PyTorch e do CTranslate2 **são compilados com `gfx1102`** [V] → começar **sem** `HSA_OVERRIDE_GFX_VERSION`; usar `11.0.0` só se alguma lib (MIOpen/hipBLASLt/rocBLAS) reclamar.

---

## Ambiente ROCm

### Estado atual da máquina [V]

| Item | Valor |
|------|-------|
| GPU | `gfx1102` (RX 7600, Navi 33) — `rocminfo` enxerga |
| Pacotes ROCm do sistema | Pacotes **do Debian** (não da AMD): `rocminfo 6.1.2`, `libamdhip64-5 5.7.1`, `librocblas0 5.5.1`, `libhsa-runtime64-1 6.1.2`. **Não existe `/opt/rocm`** |
| `/dev/kfd` | presente, grupo `render`; usuário `duck` está em `render` e `video` |
| Ollama | 0.32.5, rodando com **Vulkan** (`OLLAMA_VULKAN=1`, `OLLAMA_LLM_LIBRARY=vulkan`), modelos em `/home/duck/SSD/Extras/Models` |
| Python sistema | 3.13.5; `uv` 0.11.29 |
| Disco | `/` 27 GB livres · `/home` 31 GB livres · SSD `/home/duck/SSD/Extras` (btrfs) **230 GB livres** |

### Estratégia

- **PyTorch:** os wheels `+rocmX.Y` do PyTorch **embutem o runtime ROCm** (HIP, rocBLAS, MIOpen, hipBLASLt…). Só precisam do driver de kernel (`amdgpu`/KFD), que já está ok. Os pacotes ROCm do Debian (5.7/6.1) **não são usados** por eles.
- **CTranslate2 ROCm** (para faster-whisper/WhisperX) é diferente: o wheel é gerado com `auditwheel repair --exclude "/opt/rocm/lib/lib*"` [V], ou seja, **espera ROCm 7.2 instalado em `/opt/rocm`**. Opções: (a) instalar o userspace ROCm 7.2 da AMD (repo oficial; Debian 13 aparece nas matrizes de compatibilidade 7.x de data center [NV para Radeon]); (b) apontar `LD_LIBRARY_PATH` para as libs embutidas em `torch/lib` de um torch `+rocm7.2` [NV — gambiarra, testar]; (c) rodar o faster-whisper em CPU int8.

### Índices e versões de wheel [V em 2026-10-09]

| Índice | torch mais novo | torchaudio mais novo | Python |
|--------|-----------------|----------------------|--------|
| `https://download.pytorch.org/whl/rocm7.2` | 2.14.1 | **2.11.0** | cp312–cp315 |
| `https://download.pytorch.org/whl/rocm7.14` | 2.14.1 | — (nenhum) | cp313+ |
| `https://download.pytorch.org/whl/rocm7.1` | 2.13.0 | 2.11.0 | cp312+ |
| `https://download.pytorch.org/whl/rocm6.4` | 2.9.1 (tem 2.8.0) | 2.9.1 (tem 2.8.0) | cp311–cp314 |

**Baseline recomendado para quase todos os venvs:** `torch==2.11.0+rocm7.2`, `torchaudio==2.11.0+rocm7.2`, `torchcodec==0.11.*` (do PyPI, CPU; tabela de compatibilidade: torchcodec 0.11 ↔ torch 2.11 [V]).
**Exceção WhisperX:** `torch==2.8.0+rocm6.4`, `torchaudio==2.8.0+rocm6.4`, `torchcodec>=0.6,<0.8` (pins do whisperx 3.8.6 [V]).

Os build scripts do PyTorch compilam os wheels ROCm para `gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201;gfx1150;gfx1151;gfx1250` [V — `.ci/docker/manywheel/build.sh` no `main`; confirmar no wheel instalado com `torch.cuda.get_arch_list()`].

Exemplo de `pyproject.toml` por estágio (uv):

```toml
[project]
requires-python = ">=3.12,<3.14"
# triton-rocm (pytorch-triton-rocm no índice rocm6.4) precisa ser dependência direta:
# o torch ROCm depende dele, ele só existe no índice do PyTorch, e o uv só aplica
# `sources` a dependências diretas. Sem isso a resolução falha. [V 2026-10-10]
dependencies = ["torch==2.11.0", "torchaudio==2.11.0", "torchcodec==0.11.*", "triton-rocm"]

[tool.uv.sources]
torch = { index = "pytorch-rocm" }
torchaudio = { index = "pytorch-rocm" }
triton-rocm = { index = "pytorch-rocm" }

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[tool.uv]
# para pacotes com pin rígido de torch (ex.: chatterbox-tts pede torch==2.6.0)
override-dependencies = ["torch==2.11.0", "torchaudio==2.11.0"]
```

### Variáveis de ambiente

```bash
# Disco: tudo no SSD (btrfs suporta reflink; uv usa clone/hardlink e não duplica o torch entre venvs)
export UV_CACHE_DIR=/home/duck/SSD/Extras/dublador/uv-cache
export UV_LINK_MODE=clone          # btrfs reflink; 'hardlink' também serve (mesmo FS)
export HF_HOME=/home/duck/SSD/Extras/dublador/models/hf
export TORCH_HOME=/home/duck/SSD/Extras/dublador/models/torch
# GPU
# export HSA_OVERRIDE_GFX_VERSION=11.0.0   # SÓ se algo falhar com "invalid device function"/arch não suportada
export MIOPEN_USER_DB_PATH=/home/duck/SSD/Extras/dublador/models/miopen   # cache de kernels MIOpen (1ª execução é lenta)
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True    # reduz fragmentação em 8 GB [NV no ROCm]
```

### Problemas conhecidos gfx1102 / RDNA3

- **Não suportado oficialmente** na página "Radeon Linux support matrices" (ROCm 7.2.x lista "9000 & select 7000 series"; RX 7600 não aparece) [V]. O Ollama, por outro lado, lista RX 7600/7600 XT (gfx1102) como suportado e diz que **não** precisa de override [V].
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` faz o runtime tratar a placa como gfx1100. Funciona na maioria dos relatos de RDNA3, mas Navi 33 tem menos registradores/CU que Navi 31 → kernels gfx1100 podem falhar em casos raros [NV]. Por isso: **primeiro sem override**.
- hipBLASLt pode avisar "unsupported architecture" e cair para hipBLAS (só perda de desempenho) [NV].
- MIOpen compila kernels na primeira execução (convoluções do Demucs/wav2vec2/ECAPA) — primeira rodada bem mais lenta; persistir o cache.
- **`MIOPEN_FIND_MODE=FAST` é essencial** [V 2026-10-10]: no modo padrão o MIOpen faz busca exaustiva e compila kernels para **cada formato novo de entrada**. Com áudio de comprimento variável (o decodificador do TTS recebe um tamanho diferente a cada fala), o Qwen3-TTS levava RTF 23–62; com `FAST`, RTF 1,7–2,1. O runner exporta `MIOPEN_FIND_MODE=FAST` para todos os estágios (a não ser que já esteja definido).
- `flash-attn` (recomendado por Qwen3-TTS/Qwen3-ASR) não tem wheel ROCm simples para gfx11 → usar `attn_implementation="sdpa"` [NV quanto ao ganho/perda].
- Kernel **PREEMPT_RT** (`-rt-amd64`): sem relatos específicos encontrados de problemas amdgpu+KFD com RT; testar estabilidade em job longo [NV].
- `onnxruntime-gpu` é CUDA-only; para ROCm a AMD migrou para o EP MIGraphX → **evitar modelos ONNX na GPU** (MDX-Net `.onnx` rodam em CPU).

---

## 1. Separação de fonte (voz × música/efeitos)

Objetivo: extrair a **faixa de diálogo** (para ASR/diarização/referência de clonagem) e a **faixa de fundo** (música + SFX) que será remixada com a dublagem.

| Candidato | Tipo | SDR (benchmark do audio-separator, MUSDB) [V] | VRAM (est.) | ROCm | Licença | Notas |
|-----------|------|-----------------------------------------------|-------------|------|---------|-------|
| **BS-RoFormer-Viperx-1297** `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | PyTorch (MDXC/Roformer) | vocals 11,8 · **instrumental 16,5** (melhor instrumental) | 3–5 GB [NV] | ✅ PyTorch puro | código MIT; pesos UVR/viperx sem licença formal [NV] | **Default** do `audio-separator` |
| Mel-Band RoFormer Kim `vocals_mel_band_roformer.ckpt` | PyTorch | **vocals 12,6** (melhor vocal) | 3–5 GB [NV] | ✅ | idem | Usar se o stem de voz (para clonagem) sair sujo |
| Demucs `htdemucs_ft` | PyTorch | vocals 10,8 | ~3 GB [NV] | ✅ | MIT | Projeto arquivado/sem manutenção; 4 stems; 4× mais lento (bag of 4) |
| MDX23C-8KFFT-InstVoc_HQ / UVR-MDX-NET `.onnx` | ckpt / ONNX | vocals 10,4–10,6 | — | `.ckpt` ✅; `.onnx` só CPU | MIT (UVR) | Alternativa leve |

- **Pacote:** `audio-separator` 0.47.0 — Python `>=3.10, !=3.14.1` [V]; instalar com o torch ROCm (extra `[gpu]` puxa `onnxruntime-gpu` CUDA → **não usar**; instalar sem extra + `onnxruntime` CPU). O ROCm aparece como `cuda` no torch, então a detecção de GPU funciona [NV].
- **Vídeos de 1 h:** usar `--chunk_duration 300..600` (o README recomenda para >1 h) [V].
- **Recomendação:** rodar **BS-RoFormer-1297** e usar *instrumental* como fundo e *vocals* como diálogo. Opcional (qualidade): segundo passe Mel-RoFormer Kim só para obter uma voz mais limpa para referência de clonagem.
- **Cuidado:** esses modelos são treinados em **música**; efeitos sonoros (explosões, multidão) podem ir para o stem de voz ou ser atenuados no fundo. Validar com trechos de filme/série. Existe `mel_band_roformer_crowd_aufr33_viperx` para separar multidão [V], útil como pós-passe.
- Links: <https://github.com/nomadkaraoke/python-audio-separator> · <https://github.com/facebookresearch/demucs>

---

## 2. ASR em inglês com timestamps por palavra

| Candidato | Backend | ROCm/AMD | VRAM (est.) | Qualidade EN | Timestamps palavra | Licença | Python |
|-----------|---------|----------|-------------|--------------|--------------------|---------|--------|
| **WhisperX 3.8.6** (faster-whisper `large-v3` + wav2vec2 align + VAD) | CTranslate2 | ⚠ CT2 **4.7.0+ tem backend HIP oficial** (PR #1989) com wheels ROCm nos *releases* do GitHub (não no PyPI), compilados para `gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` com ROCm 7.2 [V]. Precisa de libs ROCm 7.2 no sistema [V]. Alinhamento wav2vec2 é torch (✅) | 3–5 GB fp16 + ~1,5 GB align [NV] | Referência de mercado | ✅ melhores (forced alignment wav2vec2) | BSD-2 (whisperx), MIT (faster-whisper/CT2), Apache/MIT (pesos) | `>=3.10,<3.14`; pins `torch~=2.8.0` [V] |
| Whisper `large-v3` / `large-v3-turbo` via `transformers` ou `openai-whisper` | PyTorch | ✅ (torch ROCm) | large-v3 fp16 ~4–6 GB; turbo ~3 GB [NV] | Igual ao WhisperX (mesmo modelo) | ✅ DTW em cross-attention (menos preciso que wav2vec2) | Apache-2.0 / MIT | 3.12/3.13 |
| **Qwen3-ASR-1.7B** + **Qwen3-ForcedAligner-0.6B** (jan/2026) | transformers (ou vLLM) | ✅ provável (transformers puro) [NV] | ~5 GB + ~2 GB aligner [NV] | WER EN **melhor que whisper-large-v3** na maioria dos sets (GigaSpeech 8,45 vs 9,76; CV-en 7,39 vs 9,90; TED 4,50 vs 6,84) [V, model card] | ✅ aligner até **5 min por chamada**, 11 idiomas | Apache-2.0 | 3.12 recomendado |
| whisper.cpp (`large-v3` GGUF/ggml) | Vulkan ou HIP (`GGML_HIP=ON`) | ✅ Vulkan funciona sem ROCm | ~2–4 GB (quantizado) [NV] | Igual ao Whisper | ⚠ token-level / DTW experimental — fraco para dublagem | MIT | n/a (binário) |
| NVIDIA Parakeet-TDT-0.6B-v3 / Canary-1B-v2 | NeMo | ⚠ NeMo em ROCm não é suportado oficialmente [NV] | ~2–4 GB | Muito bom em EN | ✅ nativos | CC-BY-4.0 | 3.10+ |

**Recomendação:**
- **Default: WhisperX** (é o padrão de fato para dublagem: VAD + segmentação + alinhamento wav2vec2 por palavra + integração com pyannote). Venv com `torch==2.11.0+rocm7.2`. A transcrição usa o **CTranslate2 ROCm na GPU** (`int8_float16`, lote 4; [validação](#tarefas-de-validação), #19), com volta automática para CT2 em CPU `int8`. O alinhamento wav2vec2 roda na GPU via torch.
- **Desafiante de qualidade:** Qwen3-ASR-1.7B + ForcedAligner (Apache, melhor WER EN publicado, PyTorch puro → sem o problema do CTranslate2). Exige fatiar o áudio em janelas ≤ 5 min para o aligner (dá para usar os segmentos do VAD/diarização).
- **Fallback simples:** `openai/whisper-large-v3` no `transformers` com `return_timestamps="word"`.
- Usar `condition_on_previous_text=False` / VAD para evitar alucinações em áudio longo [boas práticas Whisper].

Links: <https://github.com/m-bain/whisperX> · <https://github.com/OpenNMT/CTranslate2/releases> · <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> · <https://huggingface.co/openai/whisper-large-v3> · <https://github.com/ggml-org/whisper.cpp>

---

## 3. Diarização de falantes

| Candidato | ROCm | VRAM (est.) | Qualidade | Licença | Notas |
|-----------|------|-------------|-----------|---------|-------|
| **pyannote `speaker-diarization-community-1`** (pyannote.audio 4.0.7) | ✅ PyTorch puro [NV em gfx1102] | 1–2 GB [NV] | Melhor que 3.1 segundo pyannote; base do WhisperX recente | **CC-BY-4.0**; **gated** (aceitar termos + token HF uma vez; depois roda offline) [V] | Requer `torch>=2.8`, `torchaudio>=2.8`, `torchcodec>=0.7` e ffmpeg [V] |
| pyannote `speaker-diarization-3.1` | ✅ | 1–2 GB | Boa, mais antiga | MIT, gated [V] | Fallback se 4.x der problema |
| DiariZen (BUT, WavLM + Conformer + VBx sobre pyannote) | ✅ provável [NV] | 2–3 GB [NV] | DER estado-da-arte em vários benchmarks [NV] | código MIT; **pesos CC-BY-NC-4.0** [V] | OK para uso pessoal |
| NVIDIA Sortformer (`diar_streaming_sortformer_4spk-v2.1`) | ⚠ NeMo/ROCm [NV] | 2–4 GB | Bom, mas **máx. 4 falantes** | "other" (NVIDIA Open Model) [V]; v1 é CC-BY-NC | Limite de 4 falantes é ruim para filmes |

**Recomendação:** **community-1** como default, rodando no venv do ASR (WhisperX já integra) ou em venv próprio com o baseline torch 2.11. Passar `min_speakers/max_speakers` quando o usuário souber. Pós-processar: fundir falantes com < N segundos totais, e usar embeddings para re-associar falantes entre chunks se o vídeo for fatiado.

Links: <https://huggingface.co/pyannote/speaker-diarization-community-1> · <https://github.com/pyannote/pyannote-audio> · <https://github.com/BUTSpeechFIT/DiariZen>

---

## 4. Classificação de gênero por áudio

| Candidato | Arquitetura | Acurácia reportada | Licença | Notas |
|-----------|-------------|--------------------|---------|-------|
| **`JaesungHuh/voice-gender-classifier`** | ECAPA-TDNN + linear, VoxCeleb2 | 98,7 % (VoxCeleb1) [V card] | MIT [V] | Pequeno (~60 MB), rápido; precisa do código do repo GitHub do autor |
| `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech` | wav2vec2 XLS-R 300M | F1 0,999 (LibriSpeech) [V terceiros] | Apache-2.0 [V] | ~1,2 GB; `transformers` padrão |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` | wav2vec2 large | — | **CC-BY-NC-SA-4.0** [V] | 3 classes (**criança**/F/M) + idade → útil para escolher voz infantil |
| `moorlee/gender-voice-classifier-ecapa` | ECAPA + logreg | 99,3 % (EN/HI/TA) [V card] | Apache-2.0 | Modelo novo (jun/2026), pouco uso |
| inaSpeechSegmenter 0.8.0 | CNN (TensorFlow) | — | MIT | Segmenta fala/música/ruído + gênero; TF em CPU (evitar mais uma stack GPU) |

**Recomendação:** ECAPA (`JaesungHuh`) em **CPU ou GPU** (é leve), aplicado a **vários segmentos limpos por falante** (stem de voz, sem sobreposição) com **voto majoritário ponderado pela duração**; registrar confiança. Opcional: rodar `audeering` (NC, OK para uso pessoal) para detectar crianças. Se confiança < limiar → marcar `gender=unknown` e permitir override manual no CLI. Pitch (F0 mediano) como desempate barato.

---

## 5. Tradução via Ollama

### Situação local [V]

| Tag instalada | O que é de fato | Tamanho | Observação |
|---------------|-----------------|---------|------------|
| `gemma4:e4b-it-qat` | Gemma 4 E4B (7,5 B params totais), Q4_0 QAT, ctx 128k, capacidade `thinking` | 6,1 GB | Cabe na VRAM. Thinking é ligado pelo token `<|think|>` no system prompt; Ollama expõe `think` [NV p/ E4B — testar `"think": false`] |
| `qwen3:30b-a3b` (= `qwen3:30b`) | **Qwen3-30B-A3B-Thinking-2507** Q4_K_M (digest `ad815644918f` = tag `30b-a3b-thinking-2507-q4_K_M`) | 18 GB | ⚠ Modelo **só-raciocínio**: `think:false` / `/no_think` **não** desligam. Gasta tokens e tempo. |
| gguf `gemma-4-26B-A4B-it-qat-q4_0` em `~/.local/share/meu-harness/gguf/` | Gemma 4 26B-A4B MoE | — | Existe tag oficial `gemma4:26b-a4b-it-qat` no Ollama |

### Candidatos

| Modelo (tag Ollama) | VRAM / offload | Pontos fortes | Licença |
|---------------------|----------------|---------------|---------|
| **`gemma4:e4b-it-qat`** (default provisório) | ~6–7 GB, 100 % GPU | Rápido, multilíngue, já instalado | Gemma Terms of Use |
| **`qwen3:30b-a3b-instruct-2507-q4_K_M`** (opção qualidade; **precisa `ollama pull`**) | ~19 GB → offload parcial CPU; MoE 3B ativos ainda é usável | Sem modo thinking, bom em instruções/JSON | Apache-2.0 |
| `gemma4:26b-a4b-it-qat` | ~15–16 GB → offload parcial [NV] | MoE 4B ativos; provavelmente melhor que E4B | Gemma ToU |
| `translategemma:12b` (8,1 GB) / `:4b` (3,3 GB) | 12b ≈ limite da VRAM; 4b folgado | Especializado em tradução (jan/2026); en→pt-BR melhor que Gemma 3 do mesmo porte em MetricX [V paper] | Gemma ToU |

Ollama: suporta RX 7600 tanto via **ROCm** (gfx1102 listado, sem override) quanto **Vulkan** (atual) [V]. Comparar os dois backends no benchmark (tok/s e estabilidade).

**Configuração de requisição recomendada:** `format` = JSON Schema (structured outputs), `temperature` 0.2–0.3, `seed` fixo, `num_ctx` ≥ 8192, `think: false` (Gemma) e **`keep_alive: 0` no último request** do estágio. Para TranslateGemma, respeitar o template de prompt próprio (JSON pode ser mais fraco [NV]).

### Protocolo de benchmark do tradutor

1. **Corpus:** 3–4 vídeos reais (diálogo rápido de série, documentário/narração, vídeo técnico/YouTube, conteúdo com gírias). Extrair ~150 segmentos já transcritos (saída real do estágio 2), com contexto (±3 segmentos), duração do segmento original e falante.
2. **Restrição de tamanho:** para cada segmento calcular `max_chars ≈ duração_s × taxa_PT` (começar com ~15 caracteres/s, calibrar com o TTS escolhido: medir caracteres/s médio do TTS em PT-BR) e pedir no prompt + campo no schema.
3. **Saída exigida (JSON Schema):** `[{ "id": int, "pt": str, "chars": int }]`, lotes de 10–20 segmentos com glossário de nomes próprios.
4. **Modelos × backends:** `gemma4:e4b-it-qat`, `qwen3:30b-a3b-instruct-2507`, `gemma4:26b-a4b-it-qat`, `translategemma:12b`; Ollama Vulkan vs ROCm para o melhor e o mais rápido.
5. **Métricas:**

| Métrica | Como medir |
|---------|-----------|
| Fidelidade | Reference-free: **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`, gated, CC-BY-NC — ok uso pessoal) ou **MetricX-24 QE**; + revisão humana cega de 40 segmentos (erros de sentido: 0/1/2) |
| Naturalidade PT-BR | Nota humana 1–5 cega (ordem aleatória dos modelos) em 40 segmentos; checar "português de Portugal" (ex.: "estás", "autocarro", gerúndio ausente) por lista de palavras |
| Aderência a tamanho/duração | % segmentos com `len(pt) ≤ max_chars`; depois de sintetizar com o TTS: % com razão `dur_tts/dur_orig` ∈ [0,85; 1,15] e fator de stretch necessário |
| Validade JSON | % respostas que parseiam e batem com o schema na 1ª tentativa; nº de retries; ids faltando/duplicados |
| Consistência | Nomes/termos do glossário preservados; tratamento (você/tu) consistente por personagem |
| Velocidade | tok/s (prompt e geração), tempo total estimado para 1 h de vídeo, pico de VRAM/RAM |

6. **Critério de decisão:** eliminar quem tiver JSON válido < 98 % ou aderência de tamanho < 85 %; entre os restantes, ponderar fidelidade (40 %), naturalidade (35 %), aderência (15 %), velocidade (10 %). Re-rodar com 2 seeds para ver variância.

---

## 6. TTS com clonagem zero-shot em PT-BR

Ponto central para dublagem: **controle de duração** (encaixar a fala no tempo do original) e **sotaque**: clonar a voz a partir de áudio **em inglês** tende a transferir sotaque inglês para o PT (OmniVoice e Chatterbox documentam isso [V]; Chatterbox recomenda `cfg_weight=0` nesse caso).

| Modelo | Params / VRAM (est.) | PT-BR | Clonagem | Controle de duração | ROCm | Licença | Python / instalação |
|--------|---------------------|-------|----------|---------------------|------|---------|---------------------|
| **Qwen3-TTS-12Hz-1.7B-Base** (jan/2026) | ~2 B, bf16; ~4–6 GB [NV] | "Portuguese" entre 10 idiomas; variante (BR/PT) não especificada [V] | 3 s de referência + transcrição (`ref_text`); cross-lingual suportado [V] | ❌ (só via `max_new_tokens`; usar time-stretch) | ✅ provável (transformers; sem flash-attn) [NV] | **Apache-2.0** | `pip install qwen-tts` (pin `transformers==4.57.3`); Python 3.12 recomendado |
| **OmniVoice** (k2-fsa, abr/2026) | 0,6 B (base Qwen3-0.6B), fp16; ~2–3 GB [NV]; RTF até 0,025 | 646 idiomas, **16 855 h de português** (sem distinção BR/PT) [V] | ref_audio + ref_text [V] | ✅ **`duration=` em segundos** e `speed=` [V] | ✅ provável (torch puro) [NV] | código Apache-2.0; **pesos CC-BY-NC** [V] | `pip install omnivoice`; docs usam torch 2.8 |
| **Chatterbox Multilingual V3 + finetune `ResembleAI/Chatterbox-Multilingual-pt-br`** | 0,5 B; ~4 GB [NV relato] | ✅ **finetune dedicado pt-BR** [V] | ~5–10 s de referência; `exaggeration`, `cfg_weight` [V] | ❌ (ritmo indireto via cfg/exaggeration) | ✅ provável [NV] | **MIT**; saída com **marca d'água Perth** [V] | `chatterbox-tts` 0.1.7 fixa `torch==2.6.0` para Py<3.14 → **override** necessário [V] |
| XTTS-v2 (fork `coqui-tts` da Idiap) | ~0,5 B; ~3–4 GB [NV] | ✅ `pt` entre 17 idiomas; qualidade "ok", às vezes sotaque europeu [NV] | 6 s de referência | ⚠ `speed=` (sem alvo exato) | ✅ (README cita ROCm) [V] | **Coqui Public Model License (NC)** | `coqui-tts` 0.27.5, Python 3.10–3.14 [V]; torch instalado à parte |
| F5-TTS + `firstpixel/F5-TTS-pt-br` | 0,3 B; ~3 GB [NV] | ✅ finetune pt-BR (~330 h Common Voice etc.) [NV] | ref + transcrição | ✅ `fix_duration` / `speed` [NV] | ✅ | **CC-BY-NC-4.0** [V] | `f5-tts` 1.1.22 |
| Fish Audio **S2 Pro** (mar/2026) | ~5 B (4 B + 0,4 B) → **não cabe em 8 GB em bf16** | pt "Tier 2" de 80+ idiomas [V] | ✅ | ❌ | [NV] | Fish Audio Research License (NC) [V] | Só com quantização [NV] |
| Voxtral TTS (Mistral, mar/2026) | 4 B → ~8 GB+ | ✅ pt entre 9 idiomas [V] | 3 s | ❌ | [NV] (vLLM) | CC-BY-NC-4.0 [V] | Pesado demais para 8 GB |
| Kyutai **Pocket TTS** | 0,1 B, **CPU** (~6× tempo real) | ✅ pt (6 idiomas, atualização recente) [V] | ✅ (`--voice wav`) | ❌ | n/a (CPU) | CC-BY-4.0, gated [V] | `pocket-tts`, Python 3.10–3.14 — bom **fallback CPU** |
| Kokoro-82M | 0,08 B | ✅ 3 vozes PT-BR (`pf_dora`, `pm_alex`, `pm_santa`) [NV nomes e qualidade] | ❌ | `speed=` | ✅/CPU | Apache-2.0 | `kokoro` 0.9.4 exige **Python <3.13** [V] |
| CosyVoice3 (Fun-CosyVoice3-0.5B) | 0,5 B | ❌ **sem português** [V] | ✅ | instruções | — | Apache-2.0 | descartado |
| IndexTTS-2 | — | ❌ só zh/en [V] | ✅ | ✅ (duração) | — | — | descartado (seria ideal se tivesse PT) |
| Higgs Audio v2, VibeVoice | 3 B+ / 1,5 B+ | foco en/zh [NV] | ✅ | ❌ | — | — | descartados para 8 GB / PT |

### 6.1 Pesquisa ampliada (2026-10-09)

Uma segunda rodada de busca cobriu ~30 modelos (lançamentos de 2025–2026, finetunes PT-BR da comunidade, leaderboards). **Não há evidência de um modelo aberto claramente melhor que o Qwen3-TTS em PT que caiba em 8 GB**, mas surgiram concorrentes fortes, todos Apache-2.0.

Números autodeclarados no set **MiniMax-Multilingual, linha "Portuguese"** (referência PT → saída PT; ninguém publica EN→PT):

| Modelo | WER PT ↓ | SIM PT ↑ | Cabe em 8 GB | Duração-alvo | Licença |
|---|---|---|---|---|---|
| Fish Audio S2 (referência) | **1,14** | 78,1 | ❌ (mín. oficial 12 GB; fork NF4 [NV]) | ❌ | NC |
| **Qwen3-TTS-12Hz-1.7B-Base** | 1,53 | 81,7 | ✅ | ❌ | Apache-2.0 |
| **dots.tts** (jun/2026; MF / SOAR) | 1,64 | 83,1 | ✅ MF 5,7 GB / ⚠ SOAR 7,9 GB (< 41 s) [V] | ❌ | Apache-2.0 |
| **FireRedTTS3-Base** (ago/2026) | 1,79 | **86,3** | ⚠ apertado (int8 com pico de 8,3 GiB [NV]) | velocidade 0,5–2,0 (variante Instruct) | Apache-2.0 |
| **VoxCPM2** (abr/2026) | 1,94 | 83,7 | ⚠ "~8 GB" [V] | ❌ (tag de estilo) | Apache-2.0 |
| OmniVoice | 2,51 | 85,9 | ✅ | ✅ `duration=` | pesos NC |
| **MOSS-TTS-Local 1.7B v1.0** | sem número PT | — | ⚠ "3B" bf16 [NV] | ✅ **`tokens=`** (~12,5 tokens/s) [V] | Apache-2.0 |

Leitura:
- Os números vêm dos próprios autores, cada um com o seu ASR. Um benchmark independente (Gradium, concorrente comercial, com conflito de interesse) mediu WER PT **3,96** para o Qwen3-TTS, o pior de 6 sistemas. **O bake-off na máquina é obrigatório.**
- **SIM alto pode significar mais vazamento de sotaque** quando a referência é em inglês [NV]. O FireRedTTS3 e o Chatterbox recomendam referência no idioma-alvo [V].
- Agora há **dois** modelos com duração-alvo e PT: OmniVoice (`duration=`) e MOSS-TTS (`tokens=`). Eles cobrem o passo 2 do time-fit em [ARCHITECTURE.md](ARCHITECTURE.md).

**Descartados nesta rodada:**
- **Sem português:** IndexTTS-2/2.5, CosyVoice3 (a 3.5 tem PT, mas é só API), Step-Audio-EditX, FireRedTTS-2, Orpheus multilingual, Kyutai TTS, NeuTTS, CuteTTS, Kani-TTS-2 e Zonos v0.1. De memória [NV]: Spark-TTS, MegaTTS3, MaskGCT, GPT-SoVITS, Marco-Voice, SoulX-Podcast, Sesame CSM e Higgs v2.
- **Grandes demais ou só NVIDIA:** Higgs Audio v3 (~5B, NC), Voxtral, MOSS 8B em bf16 e ZONOS2 (Mini-SGLang, só NVIDIA).
- **Outros:** VibeVoice (código de TTS removido); Dia pt-BR (perdeu expressividade); OpenVoice v2 (a base não tem PT).

### 6.2 Shortlist do bake-off (provisória)

| # | Modelo | Por que está na lista |
|---|---|---|
| 1 | **Qwen3-TTS-12Hz-1.7B-Base** | Baseline e default provisório: melhor WER PT entre os abertos que cabem em 8 GB, há relatos em ROCm RDNA3.5/4 [NV] |
| 2 | **FireRedTTS3-Base** | Maior SIM PT publicado; risco de VRAM e de ser muito novo |
| 3 | **MOSS-TTS-Local 1.7B v1.0** | Duração-alvo nativa, ideal para o time-fit; sem número PT, pin `torch 2.9.1+cu128` → override |
| 4 | **OmniVoice** | `duration=` nativo e muito rápido (permite N candidatos); pior WER da lista |
| 5 | **Chatterbox ML V3 + finetune pt-br** | Único finetune BR dedicado de um laboratório; controle "PT-BR nativo"; `cfg_weight=0` contra sotaque |
| 6 | **dots.tts MF** | WER/SIM fortes e VRAM confortável (5,7 GB); alternativa: VoxCPM2 |

**Reservas:** F5-TTS-pt-br (`fix_duration`, NC), XTTS-v2 (baseline legado; existe build `AMD-PAVS-AI/XTTS-v2`), Pocket TTS e Kokoro (fallback CPU).

**Protocolo** (complementa as [Tarefas de validação](#tarefas-de-validação)):
- mesmas frases PT-BR e o **mesmo ASR** para todos (WER), SIM ECAPA contra a referência e **teste cego de sotaque BR** feito pelo usuário (o SIM sozinho pode premiar sotaque inglês);
- dois regimes de referência: **(a) EN direta** e **(b) PT "ponte"** (gerada por VoiceDesign ou vinda do banco do §7);
- hipótese a testar [NV]: TTS PT-BR nativo seguido de **conversão de voz** para o timbre do falante (Seed-VC, VC do Chatterbox, conversor do OpenVoice v2), como alternativa contra o sotaque.

**Recomendação atual:** **OmniVoice** com referência curta ([ADR-0017](adr/0017-tts-omnivoice-e-referencia-curta.md)). O resultado do bake-off está nas Tarefas de validação. Para as vozes nativas do banco (§7), a referência continua sendo de 8–15 s, um só falante, sem música e normalizada a −20 LUFS.

Fontes desta rodada: Qwen3-TTS [paper](https://arxiv.org/html/2601.15621v1) · [FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3) ([paper](https://arxiv.org/abs/2608.17492), [int8](https://huggingface.co/drbaph/FireRedTTS3-int8)) · [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) ([Local](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer)) · [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) · [dots.tts](https://github.com/rednote-hilab/dots.tts) · [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) · [Chatterbox pt-br](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br) · [Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro) · [Higgs v3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) · [ZONOS2](https://huggingface.co/Zyphra/ZONOS2) · [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) · [Gradium WER benchmark](https://gradium.ai/content/tts-wer-benchmark-2026) · [set MiniMax-Multilingual](https://arxiv.org/pdf/2505.07916) · Qwen3-TTS em ROCm: [Strix Halo](https://tinycomputers.io/posts/qwen-tts-on-amd-strix-halo.html), [qwen3-tts-hip](https://github.com/piedshag/qwen3-tts-hip).

---

## 7. Banco de vozes de fallback (por gênero)

Usado quando o falante tem pouca fala limpa (< ~6 s), sobreposição, ou o usuário desativa a clonagem.

| Fonte | Licença | Prós | Contras |
|-------|---------|------|---------|
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** — gerar vozes PT-BR por descrição ("homem adulto, voz grave, sotaque paulista neutro…") e salvar como referência | Apache-2.0 (modelo); áudio gerado é seu | Sem pessoa real; controle de gênero/idade; casa com o TTS default | Sotaque BR depende do prompt [NV] |
| **CML-TTS (pt)** — audiolivros LibriVox | **CC-BY-4.0** [V] | Leitura limpa, muitos falantes | Mistura BR/PT; leitura "de audiolivro"; atribuição exigida |
| **Multilingual LibriSpeech (pt)** | **CC-BY-4.0** [V] | 26 H / 16 M no treino (rótulos de gênero estimados por classificador) [V] | Poucos falantes; mistura de sotaques |
| Mozilla Common Voice (pt) | **CC0** | Muitos falantes BR, metadado de gênero/sotaque | Qualidade de microfone variável |
| Kokoro voices (`pf_dora`, `pm_alex`, `pm_santa`) | Apache-2.0 | Prontas, sem curadoria | Só 3 vozes; sem clonagem |
| Chatterbox/XTTS speakers embutidos | MIT / CPML (NC) | Prontos | Majoritariamente vozes EN |

**Recomendação:** montar `voices/pt-BR/{male,female,child}/NN.wav` (+ `.txt` com transcrição) com **4–6 vozes por gênero**: 2–3 geradas por VoiceDesign + 2–3 curadas de CML-TTS/Common Voice (escolher falantes claramente BR, ~10 s, limpos). Guardar `SOURCES.md` com origem e licença de cada arquivo. Atribuição por falante: escolher a voz do banco mais próxima em pitch (F0) e embedding ECAPA do falante original, sem repetir voz entre personagens do mesmo vídeo.

---

## 8. Time-stretch

| Ferramenta | Qualidade | Notas |
|------------|-----------|-------|
| **Rubber Band** CLI (`rubberband -t <ratio> --fine` / motor R3) + `pyrubberband` | Melhor para voz, preserva formantes (`--formant`) | `apt install rubberband-cli`; GPL-2.0 (ok, chamado como subprocesso). `pyrubberband` 0.4.0 (ISC) só chama o CLI |
| `ffmpeg -af atempo=X` (ou `rubberband` filter do ffmpeg, se compilado com librubberband) | Boa até ~±20 %; WSOLA | Já instalado; `atempo` aceita 0,5–100 por filtro (encadear) |
| `librosa.effects.time_stretch` | Phase vocoder, artefatos "metálicos" | Evitar para voz |

**Regra:** só aplicar stretch em [0,80; 1,25]; fora disso, voltar ao LLM pedindo versão mais curta/longa (loop limitado a 2 tentativas) ou deixar o áudio sobrepor levemente a próxima fala/silêncio. O ffmpeg local **já tem o filtro `rubberband`** [V] (`-af rubberband=tempo=X`), então dá para usar Rubber Band sem dependência Python; o CLI `rubberband` **não** está instalado [V].

---

## Orçamento de VRAM por estágio (8 GB, sequencial)

Valores estimados [NV] — medir com `rocm-smi --showmeminfo vram` / `torch.cuda.max_memory_allocated()`.

| Estágio | Modelo default | VRAM pico estimada | Precisão | Cabe? |
|---------|---------------|--------------------|----------|-------|
| 1 Separação | BS-RoFormer-1297 (chunks) | 3–5 GB | fp32/fp16 | ✅ |
| 2 ASR | faster-whisper large-v3 (CT2 ROCm) | 3–5 GB | fp16 / int8_float16 | ✅ |
| 2b Alinhamento | wav2vec2 (WhisperX) | 1–2 GB | fp32 | ✅ |
| 2-alt | Qwen3-ASR-1.7B + aligner 0.6B | 5–7 GB (carregar um de cada vez: 4–5 + 2) | bf16 | ✅ apertado |
| 3 Diarização | pyannote community-1 | 1–2 GB | fp32 | ✅ |
| 4 Gênero | ECAPA (Jaesung) | < 0,5 GB | fp32 | ✅ (ou CPU) |
| 5 Tradução | gemma4:e4b-it-qat | ~6–7 GB (pesos 6,1 GB + KV q8_0) | Q4_0 | ✅ |
| 5-alt | qwen3:30b-a3b-instruct-2507 | 8 GB GPU + ~12 GB RAM | Q4_K_M | ⚠ offload (32 GB RAM ok) |
| 6 TTS | OmniVoice 0,6 B | 2,7 GB [V] | fp16 | ✅ |
| 6-alt | OmniVoice / Chatterbox / XTTS | 2–4 GB | fp16/fp32 | ✅ |

Desktop (Xorg/Wayland) consome ~0,3–0,8 GB da mesma GPU — descontar.

## Estimativa de disco (SSD)

| Item | Tamanho aprox. |
|------|----------------|
| Cache `uv` com `torch 2.11+rocm7.2` (wheel 6,2 GB [V]; descompactado ~15–20 GB [NV]) | ~25 GB |
| Segunda stack torch `2.8.0+rocm6.4` (venv WhisperX) | ~15–20 GB |
| Venvs (via clone/hardlink — quase sem custo extra no mesmo FS) | ~2–5 GB |
| ROCm 7.2 userspace em `/opt/rocm` (só se usar CT2 ROCm; fica no `/`, que tem 27 GB livres) | ~15–25 GB [NV] ⚠ |
| Separação: BS-RoFormer + Mel-RoFormer Kim | ~1,5 GB |
| ASR: faster-whisper large-v3 (3,1 GB) + wav2vec2 align (~1,2 GB) | ~4,5 GB |
| ASR alt: Qwen3-ASR-1.7B (4,7 GB) + ForcedAligner (1,8 GB) | ~6,5 GB |
| Diarização + gênero | < 1,5 GB |
| TTS: Qwen3-TTS Base (4,5 GB) + VoiceDesign (~4,5 GB) + OmniVoice (3,3 GB) + Chatterbox (~3–4 GB necessários; repo inteiro 13,9 GB) + XTTS (2,1 GB) + Kokoro (0,3 GB) | ~18–20 GB |
| LLMs Ollama já instalados (6,1 + 18 GB) + `qwen3:30b-a3b-instruct-2507` (~19 GB) + `translategemma:12b` (8,1 GB) | ~51 GB |
| Banco de vozes + datasets (CML-TTS pt completo é grande — baixar só amostras) | 1–10 GB |
| Temporários por vídeo de 1 h (WAV 48 kHz stems + segmentos TTS) | ~5–10 GB |
| **Total** | **~150–175 GB** (cabe nos 230 GB do SSD; apagar `qwen3:30b-a3b` thinking libera 18 GB) |

---

## Tarefas de validação

Executar na máquina real antes de fixar escolhas. Marcar cada item com resultado + data.

**Ambiente**
- [x] Mover `UV_CACHE_DIR`, `HF_HOME`, `TORCH_HOME` para o SSD — 2026-10-10: `config.toml` local aponta para `~/SSD/Extras/dublador/` e o `dublador setup`/runner exportam as variáveis. Com `UV_LINK_MODE=clone` (btrfs), as duas stacks de torch somam 27 GB no `du` do cache e 27 GB no dos venvs, mas o SSD perdeu só **11 GB**: o reflink funciona e os venvs quase não custam disco extra. O cache antigo em `~/.cache/uv` não foi tocado.
- [x] Venv baseline `torch==2.11.0+rocm7.2` (Python 3.12.13, HIP 7.2.26015) — 2026-10-10, **sem** `HSA_OVERRIDE_GFX_VERSION`: `is_available()` ok, `gcnArchName` = `gfx1102`, `gfx1102` em `get_arch_list()`, 7,98 GB de VRAM (7,94 livres). Matmul fp16/bf16/fp32 corretos (erro relativo < 0,3 %), ~25 TFLOPS fp16 e bf16 em 4096² depois de aquecer (a 1ª chamada é bem mais lenta); conv1d/conv2d MIOpen corretos (a 1ª conv1d levou ~13 s compilando kernels); SDPA bf16 ok (aviso: "Mem Efficient attention" é experimental no ROCm sem `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`); alocação de 6 GB ok. Override não foi necessário.
- [x] Mesmo teste com `torch==2.8.0+rocm6.4` (HIP 6.4.43482) — 2026-10-10: tudo ok sem override, desempenho igual ao 7.2 (~25 TFLOPS fp16/bf16, mesma conv1d). `torchaudio.load` 2.8 falha por falta de backend ("Couldn't find appropriate backend"): no venv do WhisperX, ler áudio pelo `torchcodec`/ffmpeg ou instalar `soundfile`.
- [x] Job de estresse de 30 min na GPU (kernel PREEMPT_RT): sem travamentos/resets do amdgpu (`journalctl -k`) — 2026-10-10 01:06–01:36: 30 min de matmul fp16 + conv1d contínuos no stack 7.2, 14 532 iterações, ritmo estável do início ao fim, sem erro nem travamento no processo; `journalctl -k` do intervalo sem nenhuma mensagem de amdgpu/KFD/reset/timeout (conferido pelo usuário).
- [x] `torchcodec` carrega com o ffmpeg 7.1 do Debian 13 — 2026-10-10: `AudioDecoder` lê WAV 48 kHz estéreo com `torchcodec` 0.11.1 (stack 7.2) e 0.7.0 (stack 6.4); `torchaudio.load` 2.11 também funciona (usa o torchcodec por baixo).

**ASR**
- [x] CTranslate2 ROCm — 2026-10-10: a opção (b) funciona. O wheel ROCm do release 4.8.2 precisa de `libamdhip64.so.7`, `libhipblas.so.3` e `libhiprand.so.1`; o `torch 2.11+rocm7.2` traz os mesmos arquivos sem o sufixo de versão (SONAMEs iguais), então basta um diretório de symlinks + `LD_LIBRARY_PATH`. É obrigatório `CT2_CUDA_ALLOCATOR=cub_caching` (com o alocador padrão: OOM, `hipErrorIllegalAddress` ou travamento). `large-v3` em lote (`BatchedInferencePipeline`, lote 8/16): **GPU fp16 RTF 0,07** vs **CPU int8 RTF 0,48**, mesmo texto. **Decisão (usuário, 2026-10-10): o M1 transcreve na CPU int8** só com pacotes do PyPI e o torch 2.8 que o WhisperX pede; a GPU é a otimização do issue #19. `faster-whisper` 1.2.1 quebra com `av` ≥ 16 fora do WhisperX (que lê o áudio com ffmpeg).
- [~] WhisperX no estágio `transcribe` (2026-10-10, sem diarização): `large-v3` CPU int8 em lote + alinhamento wav2vec2 na GPU levam ~90 s para 70 s de áudio, contando a carga dos modelos. O venv usa `torch==2.8.0+rocm6.4`, `torchvision==0.23.0` do índice rocm6.4 e anula o `triton` de CUDA que o WhisperX puxa (`override-dependencies`). A transcrição em lote devolve janelas de VAD de até 30 s, às vezes sem pontuação; o estágio as divide por frase, pausa (≥ 0,6 s) e duração máxima (12 s) usando os tempos por palavra.
- [x] **Whisper na GPU no estágio `transcribe` (#19, 2026-10-10).**
  - **Ambiente:** o venv sobe para `torch 2.11.0+rocm7.2`, `torchvision 0.26.0` e `torchcodec 0.11.1+cpu` (índice CPU do PyTorch; a wheel do PyPI é ligada ao CUDA 13), com os pins do WhisperX anulados por `override-dependencies`. O wheel ROCm do CTranslate2 4.8.2 entra no `uv.lock` como fonte de caminho relativo (`stages/transcribe/wheels/`, que o git ignora). O `dublador setup` baixa o zip do release (sha256 conferido) para `models_dir/_downloads/` e cria o link antes do `uv sync`. Assim `uv sync`/`uv run` mantêm a versão ROCm.
  - **Shim:** o estágio cria o shim de libs dentro do próprio venv e se re-executa com `LD_LIBRARY_PATH` e `CT2_CUDA_ALLOCATOR=cub_caching`.
  - **VAD:** o VAD do pyannote que o WhisperX usa **tem de ficar na CPU**. Na GPU, o MIOpen não compila o kernel de dropout do LSTM (`miopenStatusUnknownError`), o mesmo problema do Kokoro.
  - **Números no `mlgoogle`** (626 s de fala; pico de VRAM do sistema, com o desktop):

    | Configuração | Tempo | Pico de VRAM | Texto em relação à CPU |
    |---|---|---|---|
    | CPU int8, lote 8 | 264 s | — | referência |
    | GPU fp16, lote 8 | 55 s | **7,8 GB** | — |
    | GPU fp16, lote 4 | 56 s | 6,1 GB | 10 trechos diferentes, mais erros que a CPU ("brother" no lugar de "browser") |
    | GPU fp16, lote 2 | 69 s | 6,1 GB | — |
    | GPU bf16, lote 4 | 63 s | 6,1 GB | 10 trechos diferentes, inclusive palavras inventadas |
    | **GPU `int8_float16`, lote 4** | **59 s** | **5,6 GB** | **7 trechos diferentes, metade melhor que a CPU** ("SimpleML for", "Colab") |

    Limitar o cache do alocador (`CT2_CUDA_CACHING_ALLOCATOR_CONFIG`) não baixa o pico.
  - **Decisão (usuário):** a GPU com `int8_float16` e lote 4 é o padrão, com volta automática para CPU int8 se a GPU falhar.
- [ ] WhisperX completo (transcrição + align + diarização) num vídeo de 10 min: WER manual em 2 min, precisão de bordas de palavra (±100 ms) por amostragem.
- [ ] Qwen3-ASR-1.7B + ForcedAligner no mesmo trecho (sdpa, bf16): WER, VRAM, tempo; comparar com WhisperX.

**Separação / diarização / gênero**
- [~] BS-RoFormer-1297 na RX 7600 (estágio `separate`, 2026-10-10): pico de **6,3 GB** de VRAM (com o desktop); em fp32 levou 3,2× o tempo real; com `use_autocast` (fp16) cai para ~1,0× (60 s de áudio em ~70 s com a carga do modelo), sem diferença mensurável nos stems (73 dB entre fp16 e fp32 em fala sintética sobre música; SDR da voz igual, 22,7 dB). `overlap = 2` leva a ~0,6×, também sem perda nesse teste. O pacote `audio-separator` 0.47 importa `audioread` sem declará-lo, e o extra `[cpu]` traz o `onnxruntime` de CPU. Falta a avaliação de qualidade com o vídeo de referência (SFX/multidão).
- [ ] BS-RoFormer-1297 vs Mel-RoFormer Kim vs htdemucs_ft em 3 trechos (diálogo+música, diálogo+explosões, multidão): ouvir vazamento de SFX na voz e de voz no fundo; medir VRAM e tempo para 1 h com `--chunk_duration`.
- [ ] pyannote community-1 na GPU: DER aproximado num trecho anotado à mão (5 min, 3+ falantes); tempo para 1 h.
- [~] pyannote community-1 no estágio `diarize` (2026-10-10, pyannote.audio 4.0.7, torch 2.11+rocm7.2):
  - Roda **na GPU** com `torch.backends.cudnn.enabled = False`, para o MIOpen não tentar compilar o LSTM da segmentação (o problema do VAD do WhisperX e do Kokoro).
  - No `mlgoogle` (626 s): **79 s**, pico de 3,3 GB de VRAM no sistema, **2 falantes** (175 s e 305 s de fala depois da atribuição por palavra). O classificador de gênero deu os dois como masculinos (0,99).
  - Falta o DER num trecho anotado à mão.
- [ ] Classificador de gênero: acurácia por falante em ≥ 20 falantes conhecidos (incluir vozes femininas graves, masculinas agudas, criança).
- [~] Classificador de gênero no estágio `speakers` (2026-10-10). O ECAPA de `JaesungHuh/voice-gender-classifier` (MIT) foi copiado para o estágio (`ecapa_gender.py`), porque não está no PyPI, e roda na CPU com o torch 2.11 CPU.
  - Nas 8 vozes do banco, acertou **8 de 8**, com confiança ≥ 0,97.
  - O F0 mediano das vozes masculinas geradas ficou em 160–190 Hz. Isso está acima do limiar de 145 Hz do desempate, então o F0 só decide vozes graves ou agudas sem ambiguidade.
  - Ainda falta a acurácia em ≥ 20 falantes reais.
- [x] Banco de vozes PT-BR (2026-10-10): 4 vozes masculinas e 4 femininas, geradas pelo voice design do OmniVoice no `dublador setup` (cerca de 1 min no total). Descrições e seeds estão em [`voices/README.md`](../voices/README.md).

**Tradução (Ollama)**
- [x] `ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M` — 2026-10-09: instalado (18 GB). O `ollama show` lista a capability `thinking`, mas na prática a resposta vem sem raciocínio (`thinking` vazio); ~23 tok/s no Vulkan com offload. Opcionais ainda não baixados: `gemma4:26b-a4b-it-qat`, `translategemma:12b`.
- [x] Confirmar que `"think": false` realmente desliga o raciocínio no `gemma4:e4b-it-qat` — 2026-10-10: com `think: false` o campo `thinking` vem vazio e a mesma resposta usa 279 tokens (8,4 s) contra 605 tokens e 1 296 caracteres de raciocínio (14 s) com `think: true`.
- [x] Confirmar `keep_alive: 0` / `ollama stop` libera a VRAM (checar `ollama ps` e `rocm-smi`) antes do estágio de TTS — 2026-10-10: no `rocm-smi` a VRAM volta ao nível do desktop (~1 GB) em ~0,2 s após `/api/generate` com `keep_alive: 0`, mas o `ollama ps` já fica vazio antes disso e, numa execução, ainda havia 5 GB ocupados logo depois. O estágio `translate` sempre descarrega o modelo e espera ele sair do `/api/ps`; os estágios de GPU seguintes devem conferir a VRAM livre antes de carregar o modelo.
- [x] Estágio `translate` v0 com `gemma4:e4b-it-qat` (2026-10-10): 10 segmentos em ~20 s, JSON válido de primeira, PT-BR natural ("você", "placa de vídeo"). O modelo tende a responder em minúsculas quando o contexto está em minúsculas, mesmo pedindo o contrário no prompt; o estágio capitaliza o início das frases por código.
- [ ] Rodar o **protocolo de benchmark** (§5) nos 4 modelos; comparar Ollama **Vulkan vs ROCm** (tok/s) para o modelo vencedor.

**TTS**
- [~] Qwen3-TTS-12Hz-1.7B-Base no estágio `synthesize` (2026-10-10, bf16, SDPA): carrega em ~100 s na 1ª vez, pico de **5,0 GB** de VRAM, saída 24 kHz, RTF ~1,7–2,1 com `MIOPEN_FIND_MODE=FAST` (sem isso, 23–62). As falas saem ~20–40 % mais longas que a janela original (ex.: 3,9 s para 3,2 s), como esperado sem controle de duração (M3). O pacote `qwen-tts` 0.1.1 fixa `transformers==4.57.3` e avisa que o SoX não está instalado (inofensivo). Na pipeline, o pico de VRAM do sistema chegou a **7,3 GB** dos 8 GB (modelo + caches + desktop): cabe, com pouca folga. Inteligibilidade: o Whisper large-v3 em modo `pt` transcreveu a faixa dublada praticamente igual ao texto traduzido (fala sintética do espeak como referência de voz).
- [~] Qwen3-TTS: **falha grave em 2026-10-10.** No job `mlgoogle`, 2 de 4 segmentos não emitiram o token de fim e geraram até o limite (655 s de áudio, ~24 min cada). O KV cache encheu os 8 GB e o gnome-shell caiu (`Failed to pin framebuffer with error -12`). O usuário também avaliou a voz como robotizada. O modelo saiu do estágio ([ADR-0016](adr/0016-tts-padrao-chatterbox-pt-br.md)), e o estágio passou a ter um teto de geração por segmento.
- [~] Chatterbox Multilingual V3 + finetune pt-br no estágio `synthesize` (2026-10-10, fp32, commit `5de7a54a` do GitHub com `torch 2.11.0+rocm7.2`): os pesos `t3_pt_br` e `s3gen_v3` carregam nas classes do master (no S3Gen faltam só os buffers de mel do tokenizer, recalculados no `__init__`). Carrega em ~10 s, pico de **3,6 GB** de VRAM, saída 24 kHz, ~28 tokens/s (RTF ~1,4). Referência EN com `cfg_weight` 0 / 0,3 / 0,5: a fala sai 2–2,5× mais longa que a janela original no cfg 0 e um pouco menos com cfg maior (ex.: seg de 4,44 s → 11,1 / 7,8 / 7,2 s). Nenhum segmento atingiu o teto. **Ritmo:** o falante original do teste fala 21 caracteres/s; com `cfg_weight=0` o Chatterbox sai a 9,2 caracteres/s (≈0,45×, o usuário ouviu como "0,5x"). Grade testada (caracteres/s no total / só na fala): cfg 0,5 → 11,2/13,2; cfg 0,7 → 11,9/14,3; cfg 1,0 + exaggeration 0,8 → 12,1/14,5. O ritmo satura em ~14 caracteres/s de fala, que é o normal do PT-BR. O padrão ficou `cfg_weight=0.7`, e o estágio corta o silêncio no começo e no fim de cada segmento; o resto da diferença é trabalho do `fit` (M3). O usuário achou "bem melhor" que o Qwen3-TTS, mas ainda não bom.
- [x] **Bake-off rápido de 2026-10-10** (4 frases: abertura, frase técnica, bronca, frase conversada; julgamento de ouvido feito pelo usuário; WER com o Whisper large-v3 em PT). Referências: mulher BR (amostra da ResembleAI, cortada em 9,3 s), dois narradores do CML-TTS pt (LibriVox, sotaque BR não confirmado) e o palestrante em inglês (9,2 s). Os 3 modelos rodaram em venvs descartáveis com `torch 2.11.0+rocm7.2`.
  - **OmniVoice 0.2.1** (fp16): WER 0 % nas vozes nativas e na voice design, 5,5 % no clone do palestrante; pico de 2,7 GB; ~25 s por voz (voice design: ~8 s). Avaliação: nativas "MID" a "MID-BOM"; clone do palestrante com **sotaque forte**.
  - **dots.tts MF 0.3.1** (bf16, sem `optimize`): WER 1,8–3,6 % nas nativas e 12,7 % no clone; pico de **5,8 GB** no PyTorch, que com o desktop chegou a 7,6 dos 8 GB (só terminou com `set_per_process_memory_fraction`). Avaliação: sotaque forte ou ruim. Descartado.
  - **F5-TTS pt-br** (`firstpixel`, config `F5TTS_Base`): WER 5,5–18 %; pico de 1,9 GB. Precisa do `torchcodec` +cpu do índice do PyTorch (a wheel do PyPI é ligada ao CUDA 13). Avaliação: robótico. Descartado.
  - **Salvar a clonagem no OmniVoice:** com a referência do palestrante cortada em **3,4 s** em fronteira de palavra, a avaliação ficou "MID". Com 5,4 s ficou "rápido e bugado", e com `guidance_scale` 3,5 (com referência de 3,4 s ou de 9 s) ficou "bugado". Daí a ADR-0017.
  - Kokoro (`pf_dora`, `pm_alex`, `pm_santa`), direto e como referência do Chatterbox: ruim. O LSTM do Kokoro falha no MIOpen da gfx1102 (`miopenStatusUnknownError`), então ele só roda na CPU.
- [ ] Instalar e gerar as mesmas 30 frases PT-BR (com nasais, "tch/dj", números, siglas, nomes próprios em inglês) com a **shortlist do §6.2** (Qwen3-TTS, FireRedTTS3, MOSS-TTS-Local, OmniVoice, Chatterbox pt-br, dots.tts MF) — usando **a mesma referência em inglês** e **uma referência PT-BR**.
- [ ] Avaliar: sotaque BR (nota 1–5, cego), similaridade de voz (ECAPA cosine ref×saída), inteligibilidade (WER com Whisper em PT), artefatos/alucinações (palavras puladas/repetidas), VRAM, RTF.
- [ ] OmniVoice (`duration=`) e MOSS-TTS (`tokens=`): precisão da duração-alvo (erro médio em s) e naturalidade quando a duração alvo é apertada (−15 %).
- [ ] Chatterbox: funciona com override `torch==2.11.0+rocm7.2`? Testar `cfg_weight=0` para cross-lingual.
- [ ] Medir taxa de fala (caracteres/s) de cada TTS em PT-BR para calibrar `max_chars` do tradutor.
- [ ] Comparar `-af rubberband` (já disponível no ffmpeg) vs `atempo` em fatores 0,85 e 1,2.

**Licença / acesso**
- [ ] Aceitar termos HF de `pyannote/speaker-diarization-community-1` (e `kyutai/pocket-tts`, `Unbabel/wmt22-cometkiwi-da` se usados); baixar uma vez e rodar com `HF_HUB_OFFLINE=1`.

---

## Fontes

ROCm / PyTorch / ambiente
- PyTorch wheel index ROCm: <https://download.pytorch.org/whl/rocm7.2/torch/> · <https://download.pytorch.org/whl/rocm6.4/torch/>
- PyTorch build script (arch list ROCm): <https://github.com/pytorch/pytorch/blob/main/.ci/docker/manywheel/build.sh>
- PyTorch on ROCm (AMD docs): <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html>
- Radeon Linux support matrices: <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html>
- ROCm compatibility matrix: <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- AMD — PyTorch 2.9 wheel variants ROCm: <https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html>
- torchcodec (tabela de compatibilidade): <https://github.com/meta-pytorch/torchcodec>
- Ollama GPU (AMD/Vulkan): <https://docs.ollama.com/gpu>
- PyTorch forum — hipBLASLt gfx1103 override: <https://discuss.pytorch.org/t/rocm-hipblaslt-error-with-gfx1103/212721>

Separação
- audio-separator: <https://github.com/nomadkaraoke/python-audio-separator> (models-scores.json no repo)
- Demucs: <https://github.com/facebookresearch/demucs>

ASR
- CTranslate2 releases (ROCm desde 4.7.0): <https://github.com/OpenNMT/CTranslate2/releases> · build ROCm: <https://github.com/OpenNMT/CTranslate2/blob/master/python/tools/prepare_build_environment_linux_rocm.sh>
- AMD blog CTranslate2: <https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html>
- whisper-rocm (CT2 HIP comunitário): <https://github.com/davidguttman/whisper-rocm>
- WhisperX: <https://github.com/m-bain/whisperX> · <https://pypi.org/project/whisperx/>
- Whisper large-v3: <https://huggingface.co/openai/whisper-large-v3> · turbo: <https://huggingface.co/openai/whisper-large-v3-turbo>
- Qwen3-ASR: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> · <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
- whisper.cpp: <https://github.com/ggml-org/whisper.cpp>
- Parakeet TDT v3: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> · Canary 1B v2: <https://huggingface.co/nvidia/canary-1b-v2>

Diarização / gênero
- pyannote community-1: <https://huggingface.co/pyannote/speaker-diarization-community-1> · blog: <https://pyannote.ai/blog/community-1> · <https://github.com/pyannote/pyannote-audio>
- pyannote 3.1: <https://huggingface.co/pyannote/speaker-diarization-3.1>
- DiariZen: <https://github.com/BUTSpeechFIT/DiariZen> · <https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md>
- Sortformer: <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>
- Gênero: <https://huggingface.co/JaesungHuh/voice-gender-classifier> · <https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech> · <https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender> · <https://huggingface.co/moorlee/gender-voice-classifier-ecapa> · <https://github.com/ina-foss/inaSpeechSegmenter>
- Estudo robustez classificadores de gênero: <https://arxiv.org/pdf/2509.21125>

Tradução
- Ollama library: <https://ollama.com/library/qwen3/tags> · <https://ollama.com/library/gemma4/tags> · <https://ollama.com/library/translategemma/tags>
- Gemma 4 (thinking via `<|think|>`): <https://registry.ollama.ai/library/gemma4>
- TranslateGemma tech report: <https://arxiv.org/pdf/2601.09012>
- COMET/CometKiwi: <https://github.com/Unbabel/COMET> · <https://huggingface.co/Unbabel/wmt22-cometkiwi-da>

TTS
- Qwen3-TTS: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base> · blog: <https://qwen.ai/blog?id=qwen3tts-0115>
- OmniVoice: <https://huggingface.co/k2-fsa/OmniVoice> · <https://github.com/k2-fsa/OmniVoice> (docs/languages.md) · paper <https://arxiv.org/abs/2604.00688>
- Chatterbox: <https://huggingface.co/ResembleAI/chatterbox> · pt-br: <https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br> · <https://pypi.org/project/chatterbox-tts/>
- coqui-tts (Idiap): <https://github.com/idiap/coqui-ai-TTS> · <https://coqui-tts.readthedocs.io/en/latest/installation.html> · XTTS-v2: <https://huggingface.co/coqui/XTTS-v2>
- F5-TTS: <https://github.com/SWivid/F5-TTS> · pt-br: <https://huggingface.co/firstpixel/F5-TTS-pt-br>
- Fish Audio S2 Pro: <https://huggingface.co/fishaudio/s2-pro> · <https://fish.audio/blog/fish-audio-open-sources-s2/>
- Voxtral TTS: <https://mistral.ai/news/voxtral-tts/> · <https://huggingface.co/mistralai/Voxtral-4B-TTS-2603>
- Pocket TTS: <https://huggingface.co/kyutai/pocket-tts>
- Kokoro: <https://huggingface.co/hexgrad/Kokoro-82M>
- CosyVoice3: <https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512> · IndexTTS-2: <https://huggingface.co/IndexTeam/IndexTTS-2>

Vozes / datasets
- CML-TTS: <https://www.openslr.org/146/> · <https://arxiv.org/abs/2306.10097>
- Multilingual LibriSpeech: <https://openslr.org/94/> · <https://arxiv.org/pdf/2012.03411>
- Common Voice: <https://commonvoice.mozilla.org/pt/datasets>

Time-stretch
- Rubber Band: <https://breakfastquay.com/rubberband/> · pyrubberband: <https://github.com/bmcfee/pyrubberband> · ffmpeg atempo: <https://ffmpeg.org/ffmpeg-filters.html#atempo>
