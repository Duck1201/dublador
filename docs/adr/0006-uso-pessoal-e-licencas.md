# ADR-0006 — Uso pessoal; modelos NC aceitos; código Apache-2.0

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
Vários dos melhores modelos de clonagem de voz têm licença não comercial (ex.: XTTS-v2/CPML, pesos oficiais do F5-TTS/CC-BY-NC), e alguns modelos de diarização são "gated" no Hugging Face. O resultado é para uso pessoal; o código será publicado no GitHub.

## Decisão
- Modelos são escolhidos **pela qualidade**, aceitando licenças não comerciais. `docs/MODELS.md` registra a licença de cada modelo.
- O código é licenciado sob **Apache-2.0**.
- O repositório **nunca** contém pesos de modelos, vídeos ou áudios; cada usuário baixa os modelos aceitando as licenças deles (via `dublador setup`).
- O README avisa sobre licenças dos modelos, direitos autorais do conteúdo dublado e direitos de voz/imagem.

## Consequências
- Uso comercial do resultado exigiria trocar o backend de TTS (e possivelmente outros) — possível porque o TTS é um plugin trocável (ADR-0002).
- O `.gitignore` bloqueia formatos de pesos e mídia.
