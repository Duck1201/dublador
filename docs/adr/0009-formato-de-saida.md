# ADR-0009 — Saída MKV multi-faixa com legendas

- **Status:** Aceito
- **Data:** 2026-10-09

## Contexto
Substituir o áudio impede comparar com o original quando a dublagem falha em algum trecho. Transcrição e tradução já existem com timestamps, então legendas custam quase nada.

## Decisão
A saída padrão é um **MKV** com:
- vídeo original copiado sem reencodar (`-c:v copy`);
- faixa de áudio **PT-BR** (padrão) + faixa **EN** original;
- legendas **PT-BR** e **EN** (SRT), também gravadas como arquivos soltos no job.

`--format mp4` gera o equivalente em MP4 (legendas `mov_text`) para players que não lidam bem com MKV.

## Consequências
- O mux é rápido e sem perda.
- As legendas servem também para depurar a sincronia.
