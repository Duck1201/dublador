# Banco de vozes PT-BR

Vozes brasileiras usadas por todos os falantes (padrão, ADR-0020) ou, no modo `clone`, só por quem não tem um trecho limpo de pelo menos 2 s para clonar (ADR-0002, ADR-0017). O áudio não fica no git. O `dublador setup` gera as vozes com o voice design do OmniVoice e as guarda em `<models_dir>/voices/pt-BR/<gênero>/NN.wav`, com o texto falado em `NN.txt`. Depois disso, cada voz é clonada como qualquer referência.

A tabela abaixo é a fonte de verdade de `stages/synthesize/src/synthesize/voicebank.py`, e as duas precisam ficar iguais. O F0 é a mediana medida (pyin) nesta máquina. Com `speakers.voice_strategy = "bank"`, o padrão ([ADR-0020](../docs/adr/0020-vozes-nativas-do-banco-por-padrao.md)), todo falante recebe a voz do seu gênero com o tom mais próximo do dele.

| Id | Descrição (voice design) | Seed | F0 |
|---|---|---|---|
| `male_01` | male, middle-aged, very low pitch | 11 | 87 Hz |
| `male_02` | male, elderly, very low pitch | 12 | 122 Hz |
| `male_03` | male, young adult, very low pitch | 11 | 130 Hz |
| `male_04` | male, young adult, low pitch | 12 | 150 Hz |
| `male_05` | male, middle-aged, low pitch | 102 | 165 Hz |
| `male_06` | male, young adult, moderate pitch | 101 | 179 Hz |
| `female_01` | female, middle-aged, very low pitch | 11 | 207 Hz |
| `female_02` | female, middle-aged, low pitch | 204 | 246 Hz |
| `female_03` | female, young adult, low pitch | 12 | 252 Hz |
| `female_04` | female, middle-aged, moderate pitch | 202 | 290 Hz |
| `female_05` | female, young adult, moderate pitch | 201 | 293 Hz |
| `female_06` | female, young adult, high pitch | 203 | 352 Hz |

Frase falada por todas: "Olá, tudo bem? Hoje eu vou explicar uma ideia simples, passo a passo."

**Licença:** as vozes saem do OmniVoice, cujos pesos são CC-BY-NC (uso pessoal, ADR-0006). Nenhuma imita uma pessoa real.

**Escolha da voz:** por ordem de tempo de fala, cada falante recebe a voz livre do seu gênero com o tom mais próximo do dele. Um falante de gênero `unknown` concorre a todas as vozes. Uma voz só se repete no mesmo vídeo quando as do gênero acabam.

**Para regerar uma voz:** apague o `.wav` e o `.txt` dela e rode `dublador setup --only synthesize`.
