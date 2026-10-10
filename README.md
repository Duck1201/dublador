# dublador

Pipeline de IA **100% local** para dublar vídeos de **inglês para português brasileiro**, a partir de uma CLI no Linux.

> **Status:** em implementação (marco M0, fundação). Ainda não dubla nada; veja o [`plan.md`](plan.md) para o andamento.

## O que ele faz

Você aponta um vídeo; o dublador devolve o mesmo vídeo com uma faixa de áudio dublada em PT-BR:

1. **Extrai** o áudio do vídeo.
2. **Separa** a voz do fundo (música, efeitos e ambiente são preservados).
3. **Transcreve** as falas em inglês e identifica **quem fala quando** (diarização).
4. **Traduz** para PT-BR com um LLM local, respeitando o contexto do diálogo e o tempo disponível para cada fala.
5. **Sintetiza** a fala em PT-BR **clonando a voz de cada falante**; quando não há áudio limpo suficiente de um falante, usa uma voz de um banco por gênero.
6. **Ajusta o tempo** de cada fala para caber na janela original.
7. **Mixa** a dublagem com o fundo original e gera um **MKV** com áudio PT-BR (padrão) + áudio original EN e legendas PT-BR/EN — sem reencodar o vídeo.

Nenhum dado sai da sua máquina.

## Uso (planejado)

```bash
dublador run ~/Videos/palestra.mp4             # dubla e gera palestra.ptbr.mkv
dublador run                                    # abre um seletor de arquivo (fzf)
dublador run video.mp4 --review                 # pausa após a tradução para você revisar
dublador resume                                 # continua um job interrompido ou revisado
dublador run video.mp4 --translator-model qwen3:30b-a3b-instruct-2507-q4_K_M
dublador clean --done                           # apaga intermediários pesados de jobs concluídos
```

## Requisitos (alvo da v1)

- Linux, Python gerenciado por [`uv`](https://docs.astral.sh/uv/), `ffmpeg`, [Ollama](https://ollama.com/).
- GPU AMD com ROCm (desenvolvido em uma **Radeon RX 7600, 8 GB**) e ~32 GB de RAM.
- Bastante espaço em disco: stacks do `torch` ROCm, modelos e LLMs somam **mais de 100 GB** durante a fase de comparação de modelos, e cada vídeo de 1 h gera ~5–10 GB de arquivos de trabalho. Veja a estimativa em [`docs/MODELS.md`](docs/MODELS.md#estimativa-de-disco-ssd).
- Vídeos de até **1 hora**. Não há meta de velocidade: o critério é terminar sem intervenção.

Os detalhes de modelos, VRAM e configuração do ROCm estão em [`docs/MODELS.md`](docs/MODELS.md).

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Estágios da pipeline, contrato de arquivos, CLI e configuração |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Levantamento de como esse tipo de aplicação é construído |
| [`docs/MODELS.md`](docs/MODELS.md) | Modelos escolhidos por estágio, compatibilidade com ROCm, licenças e validações pendentes |
| [`docs/adr/`](docs/adr/) | Registro das decisões de arquitetura |
| [`plan.md`](plan.md) | Plano original do projeto |

## Roteiro

1. **Levantamento** de como esse tipo de aplicação funciona → [`docs/RESEARCH.md`](docs/RESEARCH.md)
2. **Escolha dos modelos locais** para o hardware alvo → [`docs/MODELS.md`](docs/MODELS.md)
3. **Implementação**

## Licença e uso responsável

O **código** deste repositório é licenciado sob [Apache-2.0](LICENSE).

Os **modelos** usados pela pipeline **não** fazem parte do repositório: são baixados por você, sob as licenças deles. Vários dos melhores modelos de clonagem de voz têm licença **não comercial** — veja a coluna de licença em [`docs/MODELS.md`](docs/MODELS.md) antes de usar o resultado para qualquer fim além do pessoal.

Dublar e redistribuir vídeos de terceiros pode violar direitos autorais, e clonar a voz de alguém sem consentimento pode violar direitos de imagem e voz. O projeto é pensado para **uso pessoal**; a responsabilidade pelo uso do resultado é de quem o gera.
