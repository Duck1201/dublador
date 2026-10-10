# ADR-0017 — TTS padrão: OmniVoice, com referência de clonagem curta

- **Status:** Aceito (substitui a [ADR-0016](0016-tts-padrao-chatterbox-pt-br.md) e ajusta a duração da referência da [ADR-0002](0002-vozes-hibridas.md))
- **Data:** 2026-10-10

## Contexto
O usuário achou a voz do Chatterbox pt-br (ADR-0016) ruim: robótica e com sotaque. Em seguida rodamos um bake-off com OmniVoice, dots.tts MF e F5-TTS pt-br, com as mesmas 4 frases e as mesmas referências (vozes nativas PT e o palestrante em inglês). O julgamento foi do usuário, de ouvido:

- **dots.tts MF:** sotaque forte ou ruim em todas as vozes. Além disso, chegou a 7,6 dos 8 GB de VRAM, contando o desktop.
- **F5-TTS pt-br:** robótico.
- **OmniVoice:** "MID" a "MID-BOM" com referências nativas. Clonando o palestrante em inglês com a referência de ~9 s, o sotaque ficou forte.

Para salvar a clonagem, testamos variações no OmniVoice. Com uma referência de **3,4 s**, o clone do palestrante subiu para "MID", o mesmo nível das vozes nativas. Referência de 5 s e `guidance_scale` 3,5 deixaram o áudio bugado.

## Decisão
- O backend padrão do `synthesize` passa a ser o **OmniVoice** (`k2-fsa/OmniVoice`, revisão fixada, fp16, `guidance_scale` 2,0, 32 passos). O Chatterbox sai do estágio, porque os dois pedem versões incompatíveis do `transformers`.
- A clonagem continua sendo o padrão, como diz a ADR-0002, mas a referência passa de 6–15 s para **3–4,5 s (alvo 3,5 s)**, cortada em fronteira de palavra. O `speakers` escolhe a janela por palavras do transcript, com fallback de 2 s. O resto da ADR-0002 (o banco de vozes por gênero como fallback) continua valendo.

## Consequências
- Os pesos do OmniVoice são **CC-BY-NC**, aceitos para uso pessoal pela ADR-0006.
- Pico de VRAM medido em ~2,7 GB. O modelo não é autoregressivo, então não existe a geração sem fim do Qwen3-TTS. O teto por segmento continua valendo para qualquer backend.
- O OmniVoice aceita duração-alvo em segundos (`duration=`). O backend declara `supports_target_duration` e repassa a duração quando o estágio a fornecer, o que deixa o caminho aberto para o time-fit do M3.
- A referência curta transfere menos sotaque, mas também menos timbre. Se a semelhança com a voz original importar mais, a alternativa registrada é a conversão de voz depois do TTS ([MODELS.md §6.2](../MODELS.md)).
