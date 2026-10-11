# ADR-0020 — Vozes nativas do banco por padrão

- **Status:** Aceito (substitui a clonagem como padrão da [ADR-0002](0002-vozes-hibridas.md) e da [ADR-0017](0017-tts-omnivoice-e-referencia-curta.md))
- **Data:** 2026-10-10

## Contexto
No aceite do M2 + M3, com cada falante clonado da própria referência em inglês, o usuário ouviu **sotaque inglês forte** e **voz trocando de tom** com frequência. As medições confirmaram:
- **Tom:** nas falas do `SPK_01`, o tom variou de 91 a 165 Hz, contra 92 Hz na referência.
- **Testes com as mesmas 12 falas:**

  | Configuração | Desvio do tom entre falas |
  |---|---|
  | Clone com referência curta (3,6 s) | 22 Hz |
  | Clone com referência longa (9 s) | 85 Hz |
  | Voz nativa do banco | 13 Hz |

O sotaque vem da referência em inglês. Referência longa piora o tom e, no bake-off da ADR-0017, piorou também o sotaque. O usuário escolheu as vozes nativas.

## Decisão
- `speakers.voice_strategy = "bank"` passa a ser o padrão: **todo falante recebe uma voz brasileira do banco**, do seu gênero, com o **tom mais próximo** do dele (F0 mediano dos trechos limpos, comparado em escala logarítmica). Não há repetição no mesmo vídeo enquanto houver vozes.
- O banco passa de 4 + 4 para **6 + 6 vozes**, cobrindo vozes graves: masculinas de 87 a 179 Hz e femininas de 207 a 352 Hz ([`voices/README.md`](../../voices/README.md)). As vozes anteriores eram todas agudas (163–194 Hz) e não casavam com os palestrantes do vídeo de referência (84–92 Hz).
- `voice_strategy = "clone"` mantém o comportamento da ADR-0017 para quem preferir o timbre original.

## Consequências
- Sem sotaque inglês e com tom estável, mas a dublagem não imita mais o timbre de cada pessoa, só o gênero e a faixa de tom.
- O `speakers` passa a ler os WAVs do banco em `<models_dir>/voices/pt-BR/` para medir o tom de cada voz; ele continua sem gerar nada.
- A referência curta continua sendo escolhida e gravada, para o modo `clone` e para uma futura conversão de voz (Seed-VC), que poderia trazer de volta o timbre com sotaque brasileiro.
