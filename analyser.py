#!/usr/bin/env python3
"""
analisar.py - Comparação objetiva de microfones para consultas médicas
(ex.: Philips SpeechMike Ambient PSM5000 vs Audio-Technica ATR4697).

Para cada gravação calcula:
  - Níveis: ruído de fundo, nível de fala, SNR estimado, pico, loudness (LUFS), % clipping
  - Transcrição (ASR) + WER contra o guião + recall de termos clínicos
  - (opcional) Qualidade estimada sem referência: STOI / PESQ / SI-SDR (torchaudio SQUIM)
  - (opcional) PESQ / STOI intrusivos contra o áudio original (modo controlado com altifalante)

Depois emparelha as gravações feitas em simultâneo pelos dois micros e compara-as
(diferença média, intervalo de confiança por bootstrap, teste de Wilcoxon).

Convenção de nomes (ou, em alternativa, um metadados.csv na pasta de gravações):
    sala_cenario_ruido_falante_guiao_take_micro.wav
    ex.: media_B_normal_f1_g2_t1_psm5000.wav

Uso rápido:
    python analisar.py --gravacoes gravacoes --guioes guioes --saida resultados
Ver LEIA-ME.md para detalhes.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import stats
from scipy.signal import correlate, resample_poly

CAMPOS = ["sala", "cenario", "ruido", "falante", "guiao", "take", "micro"]
CHAVE_PAR = ["sala", "cenario", "ruido", "falante", "guiao", "take"]
SR_MODELOS = 16000
EXTENSOES = {".wav", ".flac", ".mp3"}

# Para cada métrica: True = maior é melhor, False = menor é melhor
METRICAS = {
    "wer": False,
    "recall_termos": True,
    "snr_estimado_db": True,
    "ruido_fundo_dbfs": False,
    "clipping_pct": False,
    "squim_stoi": True,
    "squim_pesq": True,
    "squim_sisdr": True,
    "pesq_ref": True,
    "stoi_ref": True,
}

log = logging.getLogger("analisar")


# --------------------------------------------------------------------------- áudio
def carregar_audio(caminho: Path):
    x2d, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
    return x2d, x2d.mean(axis=1), sr


def para_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR_MODELOS:
        return x.astype(np.float32)
    g = gcd(sr, SR_MODELOS)
    return resample_poly(x, SR_MODELOS // g, sr // g).astype(np.float32)


def db(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-12)))


def metricas_nivel(x2d: np.ndarray, x: np.ndarray, sr: int, frame_ms: int = 50) -> dict:
    """Ruído de fundo e nível de fala estimados pela distribuição de energia por janela.
    Não precisa de saber onde está o silêncio: percentil 10 ~ ruído, percentil 95 ~ fala."""
    n = int(sr * frame_ms / 1000)
    nf = len(x) // n
    res = {
        "duracao_s": round(len(x) / sr, 2),
        "pico_dbfs": db(float(np.max(np.abs(x2d)))),
        "clipping_pct": float(np.mean(np.abs(x2d) >= 0.999) * 100),
    }
    if nf < 10:
        return res
    fr = x[: nf * n].reshape(nf, n)
    rms_db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12)
    ruido, fala = np.percentile(rms_db, 10), np.percentile(rms_db, 95)
    res.update(
        ruido_fundo_dbfs=float(ruido),
        nivel_fala_dbfs=float(fala),
        snr_estimado_db=float(fala - ruido),
    )
    try:
        import pyloudnorm as pyln

        res["loudness_lufs"] = float(pyln.Meter(sr).integrated_loudness(x))
    except Exception as e:  # pyloudnorm em falta ou áudio demasiado curto
        log.debug("LUFS não calculado: %s", e)
    return res


def blocos_com_fala(a: np.ndarray, tamanho: int, min_dbfs: float = -50.0):
    """Divide em blocos de `tamanho` amostras e ignora os quase silenciosos."""
    for i in range(0, len(a) - tamanho // 2, tamanho):
        seg = a[i : i + tamanho]
        if len(seg) >= SR_MODELOS * 2 and db(float(np.sqrt(np.mean(seg ** 2)))) > min_dbfs:
            yield i, seg


# --------------------------------------------------------------------------- texto
_PREFIXO_ORADOR = re.compile(r"^\s*[\wÀ-ÿ ]{1,20}:\s*", re.MULTILINE)


def normalizar(texto: str, remover_oradores: bool = False) -> str:
    if remover_oradores:
        texto = _PREFIXO_ORADOR.sub("", texto)
    t = unicodedata.normalize("NFC", texto.lower())
    t = re.sub(r"[^\w\s%]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def contar(frase: str, texto: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(frase) + r"(?!\w)", texto))


def recall_termos(termos: list[str], ref: str, hip: str):
    total = encontrados = 0
    falhados = []
    for termo in termos:
        tn = normalizar(termo)
        nref = contar(tn, ref)
        if not tn or nref == 0:
            continue
        nhip = contar(tn, hip)
        total += nref
        encontrados += min(nref, nhip)
        if nhip < nref:
            falhados.append(termo)
    if total == 0:
        return None, ""
    return encontrados / total, "; ".join(falhados)


def metricas_wer(ref: str, hip: str) -> dict:
    import jiwer

    if not ref:
        return {}
    o = jiwer.process_words(ref, hip if hip else "")
    return {
        "wer": float(o.wer),
        "substituicoes": o.substitutions,
        "omissoes": o.deletions,
        "insercoes": o.insertions,
        "palavras_ref": len(ref.split()),
    }


# --------------------------------------------------------------------------- ASR
class ASRFasterWhisper:
    def __init__(self, modelo: str, lingua: str, dispositivo: str, compute_type: str):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            sys.exit("faster-whisper não instalado: pip install faster-whisper")
        log.info("A carregar Whisper '%s' (%s)...", modelo, dispositivo)
        self.modelo = WhisperModel(modelo, device=dispositivo, compute_type=compute_type)
        self.lingua = lingua

    def transcrever(self, caminho: Path) -> str:
        # Sem VAD e sem condicionamento no texto anterior: queremos que diferenças de
        # captação apareçam no WER, e evitar alucinações em loop.
        segmentos, _ = self.modelo.transcribe(
            str(caminho),
            language=self.lingua,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segmentos)


def obter_transcricao(caminho: Path, pasta: Path, asr) -> str | None:
    """Usa a transcrição em cache (pasta/<nome>.txt) se existir; senão corre o ASR."""
    ficheiro = pasta / (caminho.stem + ".txt")
    if ficheiro.exists():
        return ficheiro.read_text(encoding="utf-8")
    if asr is None:
        return None
    texto = asr.transcrever(caminho)
    ficheiro.write_text(texto, encoding="utf-8")
    return texto


# --------------------------------------------------------------------------- qualidade
class Squim:
    """Estimativas de STOI / PESQ / SI-SDR sem referência (torchaudio SQUIM)."""

    def __init__(self):
        import torch
        import torchaudio

        self.torch = torch
        self.modelo = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()

    def avaliar(self, x16: np.ndarray) -> dict:
        valores = []
        for _, seg in blocos_com_fala(x16, 10 * SR_MODELOS):
            with self.torch.inference_mode():
                s, p, d = self.modelo(self.torch.from_numpy(seg).unsqueeze(0))
            valores.append((float(s), float(p), float(d)))
        if not valores:
            return {}
        m = np.mean(valores, axis=0)
        return {"squim_stoi": m[0], "squim_pesq": m[1], "squim_sisdr": m[2]}


def alinhar(ref: np.ndarray, deg: np.ndarray, max_s: int = 90):
    """Alinha por correlação cruzada (não corrige desvio de relógio ao longo do ficheiro)."""
    n = max_s * SR_MODELOS
    r, d = ref[:n], deg[:n]
    c = correlate(d, r, mode="full", method="fft")
    atraso = int(np.argmax(np.abs(c))) - (len(r) - 1)
    if atraso > 0:
        deg = deg[atraso:]
    else:
        ref = ref[-atraso:]
    m = min(len(ref), len(deg))
    return ref[:m], deg[:m], atraso / SR_MODELOS


def metricas_intrusivas(ref16: np.ndarray, deg16: np.ndarray) -> dict:
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError:
        log.warning("pesq/pystoi não instalados: métricas com referência ignoradas")
        return {}
    ref, deg, atraso = alinhar(ref16, deg16)
    p_vals, s_vals = [], []
    for i, seg_ref in blocos_com_fala(ref, 10 * SR_MODELOS):
        seg_deg = deg[i : i + len(seg_ref)]
        try:
            p_vals.append(pesq(SR_MODELOS, seg_ref, seg_deg, "wb"))
            s_vals.append(stoi(seg_ref, seg_deg, SR_MODELOS, extended=False))
        except Exception as e:
            log.debug("bloco ignorado: %s", e)
    res = {"atraso_ref_s": round(atraso, 3)}
    if p_vals:
        res.update(pesq_ref=float(np.mean(p_vals)), stoi_ref=float(np.mean(s_vals)))
    return res


def encontrar_referencia(pasta: Path | None, meta: dict) -> Path | None:
    if pasta is None:
        return None
    for nome in (f"{meta['falante']}_{meta['guiao']}", meta["guiao"]):
        for ext in EXTENSOES:
            p = pasta / (nome + ext)
            if p.exists():
                return p
    return None


# --------------------------------------------------------------------------- metadados
def listar_gravacoes(pasta: Path) -> list[tuple[Path, dict]]:
    csv_meta = pasta / "metadados.csv"
    if csv_meta.exists():
        df = pd.read_csv(csv_meta, dtype=str).fillna("")
        em_falta = set(CAMPOS + ["ficheiro"]) - set(df.columns)
        if em_falta:
            sys.exit(f"metadados.csv sem colunas: {sorted(em_falta)}")
        return [
            (pasta / r["ficheiro"], {k: r[k] for k in CAMPOS})
            for _, r in df.iterrows()
            if (pasta / r["ficheiro"]).exists()
        ]
    itens = []
    for p in sorted(pasta.rglob("*")):
        if p.suffix.lower() not in EXTENSOES:
            continue
        partes = p.stem.split("_")
        if len(partes) != len(CAMPOS):
            log.warning("Nome fora da convenção, ignorado: %s", p.name)
            continue
        itens.append((p, dict(zip(CAMPOS, partes))))
    return itens


def carregar_guioes(pasta: Path) -> tuple[dict, dict]:
    textos, termos = {}, {}
    for p in pasta.glob("*.txt"):
        if p.stem.endswith("_termos"):
            linhas = p.read_text(encoding="utf-8").splitlines()
            termos[p.stem[: -len("_termos")]] = [l.strip() for l in linhas if l.strip()]
        else:
            textos[p.stem] = normalizar(p.read_text(encoding="utf-8"), remover_oradores=True)
    return textos, termos


# --------------------------------------------------------------------------- comparação
def bootstrap_ic(d: np.ndarray, n: int = 5000, semente: int = 0):
    rng = np.random.default_rng(semente)
    medias = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def comparar(df: pd.DataFrame, micro_a: str, micro_b: str) -> pd.DataFrame:
    metricas = [m for m in METRICAS if m in df.columns]
    pa = df[df.micro == micro_a].groupby(CHAVE_PAR)[metricas].mean()
    pb = df[df.micro == micro_b].groupby(CHAVE_PAR)[metricas].mean()
    comum = pa.index.intersection(pb.index)
    if len(comum) == 0:
        log.warning("Nenhum par simultâneo encontrado entre %s e %s", micro_a, micro_b)
        return pd.DataFrame()
    pa, pb = pa.loc[comum], pb.loc[comum]

    grupos = [("global", slice(None))]
    for c in sorted(pa.index.get_level_values("cenario").unique()):
        grupos.append((f"cenario={c}", pa.index.get_level_values("cenario") == c))
    for s in sorted(pa.index.get_level_values("sala").unique()):
        grupos.append((f"sala={s}", pa.index.get_level_values("sala") == s))

    linhas = []
    for nome, filtro in grupos:
        for m in metricas:
            va, vb = pa[m][filtro].to_numpy(float), pb[m][filtro].to_numpy(float)
            ok = ~(np.isnan(va) | np.isnan(vb))
            va, vb = va[ok], vb[ok]
            if len(va) == 0:
                continue
            d = vb - va
            ic_min, ic_max = bootstrap_ic(d) if len(d) >= 3 else (np.nan, np.nan)
            p = np.nan
            if len(d) >= 6 and np.any(d != 0):
                p = float(stats.wilcoxon(d).pvalue)
            vencedor = "inconclusivo"
            if not np.isnan(ic_min) and (ic_min > 0 or ic_max < 0):
                b_melhor = (ic_min > 0) == METRICAS[m]
                vencedor = micro_b if b_melhor else micro_a
            linhas.append({
                "grupo": nome,
                "metrica": m,
                "maior_e_melhor": METRICAS[m],
                "n_pares": len(d),
                f"media_{micro_a}": va.mean(),
                f"media_{micro_b}": vb.mean(),
                f"dif_{micro_b}_menos_{micro_a}": d.mean(),
                "ic95_min": ic_min,
                "ic95_max": ic_max,
                "p_wilcoxon": p,
                "melhor": vencedor,
            })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gravacoes", type=Path, required=True, help="pasta com os WAV/FLAC/MP3")
    ap.add_argument("--guioes", type=Path, required=True, help="pasta com <guiao>.txt e <guiao>_termos.txt")
    ap.add_argument("--referencias", type=Path, help="(opcional) áudios originais do modo controlado")
    ap.add_argument("--saida", type=Path, default=Path("resultados"))
    ap.add_argument("--asr", choices=["faster-whisper", "ficheiros", "nenhum"], default="faster-whisper",
                    help="'ficheiros' = usar só transcrições já existentes em <saida>/transcricoes")
    ap.add_argument("--modelo", default="large-v3", help="modelo Whisper (ex.: large-v3, medium)")
    ap.add_argument("--lingua", default="pt")
    ap.add_argument("--dispositivo", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--compute-type", default="default", help="ex.: float16 (GPU), int8 (CPU)")
    ap.add_argument("--sem-squim", action="store_true", help="não calcular qualidade estimada (mais rápido)")
    ap.add_argument("--micros", nargs=2, metavar=("A", "B"), help="nomes dos dois micros a comparar")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    pasta_trans = args.saida / "transcricoes"
    pasta_trans.mkdir(parents=True, exist_ok=True)

    gravacoes = listar_gravacoes(args.gravacoes)
    if not gravacoes:
        sys.exit("Nenhuma gravação encontrada.")
    textos, termos = carregar_guioes(args.guioes)
    log.info("%d gravações, %d guiões", len(gravacoes), len(textos))

    asr = ASRFasterWhisper(args.modelo, args.lingua, args.dispositivo, args.compute_type) \
        if args.asr == "faster-whisper" else None
    squim = None
    if not args.sem_squim:
        try:
            squim = Squim()
        except Exception as e:
            log.warning("SQUIM indisponível (%s); use --sem-squim para esconder este aviso", e)

    linhas = []
    for i, (caminho, meta) in enumerate(gravacoes, 1):
        log.info("[%d/%d] %s", i, len(gravacoes), caminho.name)
        linha = {"ficheiro": caminho.name, **meta}
        try:
            x2d, x, sr = carregar_audio(caminho)
            linha["sr"] = sr
            linha.update(metricas_nivel(x2d, x, sr))
            x16 = para_16k(x, sr)

            if squim is not None:
                linha.update(squim.avaliar(x16))

            ref_audio = encontrar_referencia(args.referencias, meta)
            if ref_audio is not None:
                _, r, rsr = carregar_audio(ref_audio)
                linha.update(metricas_intrusivas(para_16k(r, rsr), x16))

            hip = obter_transcricao(caminho, pasta_trans, asr)
            ref_txt = textos.get(meta["guiao"])
            if hip is not None and ref_txt:
                hip_n = normalizar(hip)
                linha.update(metricas_wer(ref_txt, hip_n))
                rec, falhados = recall_termos(termos.get(meta["guiao"], []), ref_txt, hip_n)
                linha["recall_termos"] = rec
                linha["termos_falhados"] = falhados
            elif ref_txt is None:
                log.warning("Sem guião '%s' para %s", meta["guiao"], caminho.name)
        except Exception as e:
            log.error("Erro em %s: %s", caminho.name, e)
            linha["erro"] = str(e)
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    df.to_csv(args.saida / "resultados.csv", index=False)

    metricas = [m for m in list(METRICAS) + ["nivel_fala_dbfs", "loudness_lufs"] if m in df.columns]
    resumo = df.groupby(["micro", "cenario", "sala"])[metricas].mean().round(3).reset_index()
    resumo.to_csv(args.saida / "resumo_por_condicao.csv", index=False)

    micros = args.micros or sorted(df.micro.unique())
    comp = pd.DataFrame()
    if len(micros) == 2:
        comp = comparar(df, micros[0], micros[1])
        comp.round(4).to_csv(args.saida / "comparacao_emparelhada.csv", index=False)
    else:
        log.warning("Encontrados micros %s; use --micros A B para escolher dois", micros)

    try:
        with pd.ExcelWriter(args.saida / "resultados.xlsx") as xw:
            df.to_excel(xw, sheet_name="por_gravacao", index=False)
            resumo.to_excel(xw, sheet_name="resumo_por_condicao", index=False)
            if not comp.empty:
                comp.round(4).to_excel(xw, sheet_name="comparacao", index=False)
    except Exception as e:
        log.info("Excel não gerado (%s); os CSV estão em %s", e, args.saida)

    if not comp.empty:
        g = comp[comp.grupo.str.startswith(("global", "cenario"))]
        print("\n=== Comparação emparelhada (global e por cenário) ===")
        cols = ["grupo", "metrica", "n_pares", f"media_{micros[0]}", f"media_{micros[1]}", "ic95_min", "ic95_max", "melhor"]
        print(g[cols].round(3).to_string(index=False))
    print(f"\nResultados em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()