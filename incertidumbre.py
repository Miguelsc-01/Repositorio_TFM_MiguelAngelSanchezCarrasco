# Intervalos de confianza (bootstrap) del PR-AUC de cada modelo.

# Uso:
    # python incertidumbre.py
    # python incertidumbre.py --remuestreos 5000 --nivel 0.95 --figura
    # python incertidumbre.py --modelos resnet cnn_big mobilenet

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# Los 10 modelos utilizados
MODELOS_DEFECTO = ["resnet", "mobilenet", "cnn_big", "crnn", "gru", "lstm",
                   "random_forest", "cnn", "svm", "knn"]


def cargar_scores(resultados_dir, modelo):
    # Lee las etiquetas y puntuaciones del test que guardó el entrenamiento
    ruta = Path(resultados_dir) / f"scores_{modelo}.npz"
    if not ruta.exists():
        return None
    datos = np.load(ruta)
    return datos["y_true"], datos["y_score"]


def bootstrap_metrica(etiquetas, puntuaciones, metrica, n_remuestreos, nivel, rng):
    # Bootstrap por percentiles de una métrica sobre las predicciones del test.

    # No se toca nada del modelo, ya que solo se remuestrean los índices con reemplazo.
    # Devuelve la media de la nube y los dos extremos del intervalo

    n = len(etiquetas)
    valores = np.empty(n_remuestreos, dtype=np.float64)
    validos = 0
    for _ in range(n_remuestreos):
        idx = rng.integers(0, n, size=n)    # muestreo con reemplazo
        et, pu = etiquetas[idx], puntuaciones[idx]
        # si el remuestreo deja una sola clase, la métrica no
        # tiene sentido (no hay positivos o no hay negativos), así que se descarta
        if et.min() == et.max():
            continue
        valores[validos] = metrica(et, pu)
        validos += 1
    valores = valores[:validos]
    alfa = (1.0 - nivel) / 2.0  # p.ej. 0.025 para el 95%
    ic_bajo, ic_alto = np.percentile(valores, [100 * alfa, 100 * (1 - alfa)])
    return float(np.mean(valores)), float(ic_bajo), float(ic_alto)


def _guardar_markdown(filas, ruta, nivel):
    # crea la tabla en Markdown, para utilizarla en la memoria
    pct = int(round(nivel * 100))
    lineas = [f"# Intervalos de confianza del PR-AUC (bootstrap, {pct}%)", "",
              f"| Modelo | PR-AUC | IC {pct}% | ROC-AUC | IC {pct}% |",
              "| --- | --- | --- | --- | --- |"]
    for f in filas:
        lineas.append(
            f"| {f['modelo']} | {f['pr_auc']:.4f} | "
            f"[{f['pr_ic_bajo']:.4f}, {f['pr_ic_alto']:.4f}] | "
            f"{f['roc_auc']:.4f} | "
            f"[{f['roc_ic_bajo']:.4f}, {f['roc_ic_alto']:.4f}] |")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _guardar_csv(filas, ruta):
    # Crea el CSV, por si se quiere utilizar en una hoja de cálculo
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def _grafica(filas, ruta, nivel):
    # Crea la gráfica de puntos con barras de error (PR-AUC +/- IC) por modelo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ordena de peor a mejor
    filas = sorted(filas, key=lambda f: f["pr_auc"])
    nombres = [f["modelo"] for f in filas]
    medias = np.array([f["pr_auc"] for f in filas])
    err_bajo = medias - np.array([f["pr_ic_bajo"] for f in filas])
    err_alto = np.array([f["pr_ic_alto"] for f in filas]) - medias

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(medias, range(len(filas)), xerr=[err_bajo, err_alto],
                fmt="o", capsize=4)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(nombres)
    ax.set_xlabel(f"PR-AUC (test) con IC {int(nivel * 100)}% por bootstrap")
    ax.set_title("Incertidumbre del PR-AUC por modelo")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main(args):
    rng = np.random.default_rng(args.semilla)   # generador reproducible
    resultados_dir = Path(args.resultados)
    pct = int(round(args.nivel * 100))

    filas = []
    for modelo in args.modelos:
        cargado = cargar_scores(resultados_dir, modelo)
        if cargado is None:
            print(f"[aviso] no se encuentra scores_{modelo}.npz.")
            continue
        etiquetas, puntuaciones = cargado

        # Estimación puntual + su intervalo por bootstrap
        pr_auc = float(average_precision_score(etiquetas, puntuaciones))
        _, pr_lo, pr_hi = bootstrap_metrica(
            etiquetas, puntuaciones, average_precision_score,
            args.remuestreos, args.nivel, rng)

        roc_auc = float(roc_auc_score(etiquetas, puntuaciones))
        _, roc_lo, roc_hi = bootstrap_metrica(
            etiquetas, puntuaciones, roc_auc_score,
            args.remuestreos, args.nivel, rng)

        filas.append({
            "modelo": modelo, "n_test": int(len(etiquetas)),
            "pr_auc": pr_auc, "pr_ic_bajo": pr_lo, "pr_ic_alto": pr_hi,
            "roc_auc": roc_auc, "roc_ic_bajo": roc_lo, "roc_ic_alto": roc_hi,
        })
        print(f"{modelo:14s}  PR-AUC={pr_auc:.4f}  "
              f"IC{pct}%=[{pr_lo:.4f}, {pr_hi:.4f}]")

    if not filas:
        raise SystemExit("No se pudo leer ningún scores_*.npz. "
                         "¿Se ha entrenado y están en la carpeta 'results/'?")

    _guardar_markdown(filas, resultados_dir / "incertidumbre_pr_auc.md", args.nivel)
    _guardar_csv(filas, resultados_dir / "incertidumbre_pr_auc.csv")
    print(f"\nGuardado en {resultados_dir / 'incertidumbre_pr_auc.md'} (y en .csv)")

    if args.figura:
        destino = resultados_dir / "figuras" / "incertidumbre_pr_auc.png"
        _grafica(filas, destino, args.nivel)
        print(f"Figura en {destino}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Intervalos de confianza (bootstrap) del PR-AUC por modelo")
    ap.add_argument("--resultados", default="results",
                    help="carpeta donde están los scores_*.npz")
    ap.add_argument("--modelos", nargs="+", default=MODELOS_DEFECTO,
                    help="modelos a evaluar (por defecto los 10)")
    ap.add_argument("--remuestreos", type=int, default=2000,
                    help="nº de repeticiones del bootstrap")
    ap.add_argument("--nivel", type=float, default=0.95,
                    help="nivel de confianza (0.95 = intervalo del 95%%)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--figura", action="store_true",
                    help="genera además la gráfica con barras de error")
    main(ap.parse_args())
