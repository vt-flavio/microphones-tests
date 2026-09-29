# Análise de microfones: PSM5000 vs ATR4697

## Instalação

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Estrutura de pastas

    gravacoes/
        media_B_normal_f1_g2_t1_psm5000.wav
        media_B_normal_f1_g2_t1_atr4697.wav
        ...
    guioes/
        g2.txt            texto de referência do guião (prefixos "Médico:" / "Paciente:" são ignorados)
        g2_termos.txt     um termo clínico por linha (ex.: amoxicilina, 875 mg, sem febre)
    referencias/          (opcional, modo controlado) áudio original: g2.wav ou f1_g2.wav

Nome das gravações: `sala_cenario_ruido_falante_guiao_take_micro.wav`, sem "_" dentro de cada campo.
Se as gravações atuais não seguem esta convenção, crie `gravacoes/metadados.csv` com as colunas
`ficheiro,sala,cenario,ruido,falante,guiao,take,micro`.

Duas gravações formam um par (feitas em simultâneo) quando só o campo `micro` difere.

## Execução

    # completo: Whisper local + todas as métricas
    python analisar.py --gravacoes gravacoes --guioes guioes --saida resultados

    # com GPU
    python analisar.py ... --dispositivo cuda --compute-type float16

    # rápido, sem SQUIM
    python analisar.py ... --sem-squim

    # modo controlado (altifalante): acrescenta PESQ/STOI contra o original
    python analisar.py ... --referencias referencias

    # ASR externo: ponha as transcrições em resultados/transcricoes/<nome_do_wav>.txt
    python analisar.py ... --asr ficheiros

As transcrições ficam guardadas em `resultados/transcricoes/`. Em execuções seguintes são
reutilizadas, por isso só se paga o custo do Whisper uma vez.

## Resultados

- `resultados.csv`: uma linha por gravação, com todas as métricas e os termos clínicos falhados.
- `resumo_por_condicao.csv`: médias por micro × cenário × sala.
- `comparacao_emparelhada.csv`: diferença entre micros nos pares simultâneos, com IC 95% (bootstrap),
  p-valor de Wilcoxon (a partir de 6 pares) e qual é melhor. "inconclusivo" quando o IC inclui zero.
- `resultados.xlsx`: o mesmo, em folhas separadas.

## Como ler as métricas

| Métrica | Melhor | Nota |
|---|---|---|
| wer | menor | Métrica principal. 0,10 = 10% das palavras erradas |
| recall_termos | maior | % de termos clínicos transcritos corretamente |
| snr_estimado_db | maior | Fala vs ruído de fundo |
| ruido_fundo_dbfs | menor | Nível do ruído de fundo |
| clipping_pct | menor | Deve ser ~0; se não, o ganho está alto demais |
| squim_stoi / squim_pesq / squim_sisdr | maior | Estimativas por modelo, sem referência |
| pesq_ref / stoi_ref | maior | Só no modo controlado |

## Limitações

- O WER depende da normalização do texto. Escrevam números nos guiões da mesma forma que o ASR
  os escreve (normalmente em algarismos: "875 mg").
- O alinhamento com a referência corrige o atraso inicial, não o desvio de relógio ao longo do ficheiro.
  Para gravações longas, prefiram guiões de 3–5 minutos.
- Com poucos pares, o resultado tende a sair "inconclusivo". O ideal é ter pelo menos 6 pares por cenário.
- O processamento interno do PSM5000 pode melhorar as métricas SQUIM sem melhorar o WER.
  Decidam pelo WER e pelos termos clínicos.