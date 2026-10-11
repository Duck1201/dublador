# ADR-0021 — Banco sem as vozes "very low pitch"

- **Status:** Aceito (ajusta a [ADR-0020](0020-vozes-nativas-do-banco-por-padrao.md))
- **Data:** 2026-10-10

## Contexto
Com o banco da ADR-0020, os palestrantes graves do vídeo de referência ficaram com as vozes criadas como "very low pitch" (87 e 122 Hz). O usuário ouviu um resultado pior: tom ainda variando, sensação de troca de voz e fala **mais robótica**. Num A/B com as mesmas 20 falas (clone; banco grave; banco moderado, com `male_04` e `male_06` do banco anterior; clone com o tom corrigido), ele escolheu o **banco moderado**.

## Decisão
O banco fica só com descrições "low", "moderate" e "high": 4 vozes masculinas (150–196 Hz) e 5 femininas (246–352 Hz), listadas em [`voices/README.md`](../../voices/README.md). A escolha continua pelo tom mais próximo (ADR-0020). Um falante grave recebe a voz masculina mais grave que ainda soa natural, e não uma voz extrema.

## Consequências
- Falantes muito graves ficam com uma voz um pouco mais aguda que a original. É a troca aceita pela naturalidade.
- O voice design do OmniVoice não produz vozes graves naturais. Se isso fizer falta, o caminho é uma referência humana nativa grave (com licença adequada) no banco, e não a descrição.
