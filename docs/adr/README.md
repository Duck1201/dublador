# Architecture Decision Records

Decisões de arquitetura do dublador, no formato curto: **contexto → decisão → consequências**. Um ADR aceito não é editado; se a decisão mudar, um novo ADR o substitui e o antigo é marcado como `Substituído por ADR-XXXX`.

| ADR | Decisão | Status |
|---|---|---|
| [0001](0001-escopo-e-idiomas.md) | Escopo: EN→PT-BR, idiomas parametrizados | Aceito |
| [0002](0002-vozes-hibridas.md) | Vozes híbridas: clonagem + fallback por gênero | Aceito (ajustado por 0017 e 0020) |
| [0003](0003-separacao-voz-fundo.md) | Separar voz do fundo em vez de descartar o áudio original | Aceito |
| [0004](0004-sincronia-time-fit.md) | Sincronia por segmento (time-fit), sem lip-sync | Aceito (detalhado por 0018) |
| [0005](0005-traducao-por-llm-local.md) | Tradução por LLM local configurável | Aceito (ajustado por 0015) |
| [0006](0006-uso-pessoal-e-licencas.md) | Uso pessoal; modelos NC aceitos; código Apache-2.0 | Aceito |
| [0007](0007-venv-por-estagio-e-contrato-de-arquivos.md) | Um venv por estágio, contrato de arquivos, execução sequencial | Aceito |
| [0008](0008-revisao-opcional.md) | Revisão humana opcional após a tradução | Aceito |
| [0009](0009-formato-de-saida.md) | Saída MKV multi-faixa com legendas | Aceito |
| [0010](0010-cli.md) | CLI Typer com seletor fzf opcional | Aceito |
| [0011](0011-armazenamento.md) | Código em /home, modelos e trabalho no SSD, tudo configurável | Substituído (0014) |
| [0012](0012-idioma-da-documentacao-e-codigo.md) | Documentação em PT-BR, código em inglês | Aceito |
| [0013](0013-escopo-de-validacao-v1.md) | V1 validada com vídeos curtos, suporte até 1 h | Aceito |
| [0014](0014-tudo-no-ssd.md) | Cache do uv, venvs e modelos no SSD; só o código no /home | Aceito |
| [0015](0015-tradutor-qualidade-instruct.md) | Opção de qualidade do tradutor: Qwen3 30B-A3B Instruct | Aceito |
| [0016](0016-tts-padrao-chatterbox-pt-br.md) | TTS padrão: Chatterbox Multilingual V3 com finetune pt-BR | Substituído (0017) |
| [0017](0017-tts-omnivoice-e-referencia-curta.md) | TTS padrão: OmniVoice, com referência de clonagem curta (3–4,5 s) | Aceito (clonagem deixa de ser o padrão: 0020) |
| [0018](0018-ordem-do-time-fit-e-loop.md) | Ordem do ajuste de tempo e loop de encurtamento | Aceito (ajustado por 0019) |
| [0019](0019-desacelerar-falas-curtas.md) | Desacelerar levemente as falas que ficam curtas (até 0,85×) | Aceito |
| [0020](0020-vozes-nativas-do-banco-por-padrao.md) | Vozes nativas do banco por padrão, escolhidas pelo tom | Aceito (ajustado por 0021) |
| [0021](0021-banco-sem-vozes-muito-graves.md) | Banco sem as vozes "very low pitch" (robóticas) | Aceito |
