# Banco de vozes PT-BR

Vozes de fallback, usadas quando um falante não tem um trecho limpo de pelo menos 2 s para clonar (ADR-0002, ADR-0017). O áudio não fica no git. O `dublador setup` gera as vozes com o voice design do OmniVoice e as guarda em `<models_dir>/voices/pt-BR/<gênero>/NN.wav`, com o texto falado em `NN.txt`. Depois disso, cada voz é clonada como qualquer referência.

A tabela abaixo é a fonte de verdade de `stages/synthesize/src/synthesize/voicebank.py`, e as duas precisam ficar iguais. O `speakers` escolhe só o id (`bank_voices_per_gender = 4`).

| Id | Descrição (voice design) | Seed |
|---|---|---|
| `male_01` | male, young adult, moderate pitch | 101 |
| `male_02` | male, middle-aged, low pitch | 102 |
| `male_03` | male, young adult, low pitch | 103 |
| `male_04` | male, middle-aged, moderate pitch | 104 |
| `female_01` | female, young adult, moderate pitch | 201 |
| `female_02` | female, middle-aged, moderate pitch | 202 |
| `female_03` | female, young adult, high pitch | 203 |
| `female_04` | female, middle-aged, low pitch | 204 |

Frase falada por todas: "Olá, tudo bem? Hoje eu vou explicar uma ideia simples, passo a passo."

**Licença:** as vozes saem do OmniVoice, cujos pesos são CC-BY-NC (uso pessoal, ADR-0006). Nenhuma imita uma pessoa real.

**Escolha da voz:** um falante sem referência recebe a próxima voz livre do seu gênero, por ordem de tempo de fala. Um falante de gênero `unknown` alterna entre masculino e feminino. Uma voz só se repete no mesmo vídeo quando as do gênero acabam.

**Para regerar uma voz:** apague o `.wav` e o `.txt` dela e rode `dublador setup --only synthesize`.
