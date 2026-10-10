# ADR-0012 — Documentação em PT-BR, código em inglês

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
O público natural de uma ferramenta EN→PT-BR fala português, e documentação que o mantenedor não tem vontade de atualizar envelhece rápido. Código em inglês é a convenção universal.

## Decisão
- **Documentação** (README, `docs/`, ADRs): PT-BR.
- **Código** (identificadores, comentários, docstrings), **commits**, mensagens de PR, **mensagens de CLI e de erro**: inglês.

## Consequências
- Prompts do tradutor podem ser em inglês ou português — decisão técnica do estágio `translate`, não desta regra.
