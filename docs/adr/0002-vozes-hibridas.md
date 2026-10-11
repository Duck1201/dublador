# ADR-0002 — Vozes híbridas: clonagem + fallback por gênero

- **Status:** Aceito (duração da referência ajustada pela [ADR-0017](0017-tts-omnivoice-e-referencia-curta.md); banco nativo por padrão na [ADR-0020](0020-vozes-nativas-do-banco-por-padrao.md))
- **Data:** 2026-10-09

## Contexto
O plano original pede que a transcrição identifique cada ator com seu gênero, para o TTS gerar uma voz por ator. Uma voz genérica por gênero é simples, mas soa artificial; a clonagem zero-shot preserva o timbre de cada ator, porém exige alguns segundos de áudio limpo do falante — o que falta para personagens que falam pouco.

## Decisão
**Clonagem zero-shot por padrão**, usando um trecho de referência (6–15 s) extraído da trilha de voz separada. Quando um falante não tem referência boa o bastante, o sistema usa uma voz de um **banco de vozes PT-BR por gênero** (`male_1`, `female_1`…), atribuída de forma estável por falante.

## Consequências
- É preciso um **classificador de gênero** por falante (estágio `speakers`), mas ele só é decisivo no fallback.
- É preciso montar e versionar os metadados de um banco de vozes de fallback com licença adequada (o áudio não vai para o git).
- A qualidade da clonagem depende diretamente da separação voz/fundo (ADR-0003) e da diarização.
- O backend de TTS é um plugin trocável, porque é a área de modelos que mais evolui.
