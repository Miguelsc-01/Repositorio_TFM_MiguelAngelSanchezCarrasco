# Análisis de resultados: tablas, figuras, benchmarks y consumo.

# Comandos:
    # python analisis.py tabla | cuantizacion | final
    # python analisis.py robustez        --config config.yaml
    # python analisis.py robustez-int8   --config config.yaml
    # python analisis.py bench-pc        --config config.yaml
    # python analisis.py bench-pi        (en la Pi)
    # python analisis.py consumo         (en la Pi)

from __future__ import annotations
import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

# Imports que solo hacen falta en el PC (tablas, figuras, entrenamiento).
# En la Raspberry Pi no están instalados y no se necesitan para 'bench-pi'/'consumo', así que si faltan se sigue la ejecución sin ellos.
try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import joblib
    import onnxruntime as ort
    from utiles import load_config, metrics_at_threshold
    from datos import waveform_to_features, fix_length, mix_noise, mfcc_statistics
    from modelos import build_model
except ImportError:
    pass

RES = Path("results")
FIG = RES / "figuras"

# Familia y si es ejecutable en NPU (para etiquetar la tabla)
INFO = {
    "svm": ("clásico", "CPU"), "random_forest": ("clásico", "CPU"),
    "knn": ("clásico", "CPU"),
    "lstm": ("recurrente", "CPU"), "gru": ("recurrente", "CPU"),
    "crnn": ("recurrente", "CPU"),
    "cnn": ("CNN", "NPU"), "cnn_big": ("CNN", "NPU"),
    "resnet": ("CNN", "NPU"), "mobilenet": ("CNN", "NPU"),
}


def _load_results():
    rows, scores = {}, {}
    for p in sorted(RES.glob("results_*.json")):
        name = p.stem[len("results_"):]
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        m = d.get("test_tuned", {})
        fam, npu = INFO.get(name, ("?", "?"))
        rows[name] = {"modelo": name, "familia": fam,
                      "PR-AUC": m.get("pr_auc"), "F1": m.get("f1"),
                      "bal_acc": m.get("balanced_accuracy"),
                      "precision": m.get("precision"), "recall": m.get("recall"),
                      "params": d.get("n_params"), "edge": npu}
        sp = RES / f"scores_{name}.npz"
        if sp.exists():
            z = np.load(sp)
            scores[name] = z["y_score"]
    return rows, scores


def _df_to_md(df):
    def fmt(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.4f}"
        return "—" if v is None else str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def tabla_modelos_main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows, scores = _load_results()
    if not rows:
        print("No hay resultados todavía (results_*.json).")
        return
    df = pd.DataFrame(rows.values()).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    df.to_csv(RES / "tabla_modelos.csv", index=False)
    (RES / "resumen_modelos.md").write_text(
        "# Comparativa de modelos (dataset, test)\n\n" + _df_to_md(df) + "\n",
        encoding="utf-8")

    # Barras PR-AUC y F1
    dd = df.melt(id_vars="modelo", value_vars=["PR-AUC", "F1"],
                 var_name="métrica", value_name="valor")
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=dd, x="modelo", y="valor", hue="métrica")
    plt.ylim(min(0.5, dd["valor"].min() - 0.05), 1.0)
    plt.xticks(rotation=30, ha="right"); plt.title("Comparativa de arquitecturas (test)")
    plt.tight_layout(); plt.savefig(FIG / "barras_metricas.png", dpi=150); plt.close()

    # Heatmap modelos x métricas
    hm = df.set_index("modelo")[["PR-AUC", "F1", "bal_acc", "precision", "recall"]].astype(float)
    plt.figure(figsize=(7, 0.5 * len(hm) + 1.5))
    sns.heatmap(hm, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "valor"})
    plt.title("Métricas por modelo"); plt.ylabel(""); plt.tight_layout()
    plt.savefig(FIG / "heatmap_metricas.png", dpi=150); plt.close()

    # Matriz de correlación entre predicciones de los modelos
    if len(scores) >= 2:
        names = list(scores.keys())
        M = np.vstack([scores[n] for n in names])
        corr = np.corrcoef(M)
        plt.figure(figsize=(0.7 * len(names) + 2, 0.7 * len(names) + 2))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1,
                    xticklabels=names, yticklabels=names)
        plt.title("Correlación entre predicciones (test)")
        plt.tight_layout(); plt.savefig(FIG / "correlacion_modelos.png", dpi=150); plt.close()

    print(df.to_string(index=False))
    print(f"\nGenerado en {RES}/resumen_modelos.md, tabla_modelos.csv y {FIG}/")



RES = Path("results")
FIG = RES / "figuras"


def _load():
    rows, per_model = [], {}
    for p in sorted(RES.glob("quantization_*.json")):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        model = d["model"]; nparams = d.get("n_params")
        d.setdefault("device", "NPU" if d.get("npu_capable", True) else "CPU")
        d.setdefault("family", "CNN" if d.get("npu_capable", True) else "recurrente")
        per_model[model] = d
        for var, r in d["results"].items():
            m = r["metrics"]
            rows.append({"modelo": model, "familia": d["family"], "device": d["device"],
                         "params": nparams, "variante": var,
                         "PR-AUC": m["pr_auc"], "F1": m["f1"],
                         "bal_acc": m["balanced_accuracy"],
                         "tam_MB": r["size_mb"], "lat_ms_cpu": r["latency_ms"]})
    return pd.DataFrame(rows), per_model


def tabla_cuant_main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, per_model = _load()
    if df.empty:
        print("No hay resultados de cuantización (quantization_*.json).")
        return
    df = df.sort_values(["params", "variante"]).reset_index(drop=True)
    df.to_csv(RES / "quant_tabla.csv", index=False)

    # Tabla markdown
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else str(v))
    cols = ["modelo", "familia", "device", "params", "variante", "PR-AUC", "F1", "bal_acc", "tam_MB", "lat_ms_cpu"]
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df[cols].itertuples(index=False):
        md.append("| " + " | ".join(fmt(v) for v in r) + " |")
    (RES / "quant_summary.md").write_text(
        "# Cuantización de las CNN (test)\n\n" + "\n".join(md) + "\n", encoding="utf-8")

    # Degradación INT8 vs FP32 por modelo
    deg = []
    for model, d in per_model.items():
        r = d["results"]
        if "int8" in r and "fp32" in r:
            deg.append({"modelo": model, "params": d.get("n_params"),
                        "familia": d.get("family", "CNN"),
                        "dPR_AUC": r["int8"]["metrics"]["pr_auc"] - r["fp32"]["metrics"]["pr_auc"],
                        "dF1": r["int8"]["metrics"]["f1"] - r["fp32"]["metrics"]["f1"],
                        "red_tam": 100 * (1 - r["int8"]["size_mb"] / r["fp32"]["size_mb"])})
    dd = pd.DataFrame(deg).sort_values("params")

    if not dd.empty:
        # Degradación vs tamaño del modelo
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.plot(dd["params"], dd["dPR_AUC"], "o-", label="Δ PR-AUC")
        ax.plot(dd["params"], dd["dF1"], "s-", label="Δ F1")
        for _, row in dd.iterrows():
            ax.annotate(row["modelo"], (row["params"], row["dF1"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("nº de parámetros (escala log)")
        ax.set_ylabel("cambio al cuantizar a INT8 (test)")
        ax.set_title("Degradación por cuantización frente al tamaño del modelo")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "degradacion_vs_tamano.png", dpi=150); plt.close()

        # Reducción de tamaño INT8 por familia
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {"CNN": "#2b6cb0", "recurrente": "#c05621"}
        bars = ax.bar(dd["modelo"], dd["red_tam"],
                      color=[colors.get(f, "#888") for f in dd["familia"]])
        ax.set_ylabel("reducción de tamaño al cuantizar a INT8 (%)")
        ax.set_title("Cuánto se deja cuantizar cada arquitectura")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=colors["CNN"], label="CNN (a NPU)"),
                           Patch(color=colors["recurrente"], label="recurrente (CPU)")])
        plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(FIG / "reduccion_tamano_familia.png", dpi=150); plt.close()

    # Tamaño por variante
    piv = df.pivot_table(index="modelo", columns="variante", values="tam_MB")
    piv = piv.reindex(sorted(piv.index, key=lambda m: per_model[m].get("n_params", 0)))
    ax = piv.plot(kind="bar", figsize=(9, 4.5))
    ax.set_ylabel("tamaño en disco (MB)"); ax.set_title("Tamaño por variante de cuantización")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(FIG / "tamano_por_variante.png", dpi=150); plt.close()

    # Plano precisión vs tamaño
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"fp32": "o", "fp16": "s", "int8": "^"}
    for var in ["fp32", "fp16", "int8"]:
        sub = df[df["variante"] == var]
        if not sub.empty:
            ax.scatter(sub["tam_MB"], sub["PR-AUC"], marker=markers[var], s=70, label=var)
    for _, row in df.iterrows():
        ax.annotate(row["modelo"], (row["tam_MB"], row["PR-AUC"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("tamaño en disco (MB)"); ax.set_ylabel("PR-AUC (test)")
    ax.set_title("Precisión frente a tamaño (por modelo y variante)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "precision_vs_tamano.png", dpi=150); plt.close()

    print(df.to_string(index=False))
    print(f"\nTabla en {RES}/quant_summary.md y figuras en {FIG}/")



RES = Path("results")
FIG = RES / "figuras"

ALL = ["cnn", "cnn_big", "resnet", "mobilenet", "crnn", "lstm", "gru", "svm", "random_forest", "knn"]
NPU = {"cnn", "cnn_big", "resnet", "mobilenet"}
RNN = {"crnn", "lstm", "gru"}


def load_json(name):
    p = RES / name
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def first(lst, **flt):
    for r in lst or []:
        if all(r.get(k) == v for k, v in flt.items()):
            return r
    return None


def comparativa_final_main():
    FIG.mkdir(parents=True, exist_ok=True)
    bench_pc = load_json("bench_pc.json") or []
    bench_pi = load_json("bench_pi.json") or []
    power_pi = load_json("power_pi.json") or []
    snr = load_json("robustez_snr_quant.json")

    def recall_at(model, snr_db=-5, variant="int8"):
        if not snr:
            return None
        grid = snr["snr_grid"]
        if snr_db not in grid:
            return None
        rec = snr["recall"].get(model, {}).get(variant)
        return rec[grid.index(snr_db)] if rec else None

    rows = []
    for m in ALL:
        rp = load_json(f"results_{m}.json")
        q = load_json(f"quantization_{m}.json")
        if rp is None and q is None:
            continue
        fam = "CNN" if m in NPU else ("recurrente" if m in RNN else "clásico")
        tuned = (rp or {}).get("test_tuned", {})
        row = {"modelo": m, "familia": fam, "params": (q or {}).get("n_params"),
               "PR_AUC_fp32": tuned.get("pr_auc"), "F1_fp32": tuned.get("f1")}
        if q:
            r = q["results"]
            row["PR_AUC_int8"] = r.get("int8", {}).get("metrics", {}).get("pr_auc")
            row["F1_int8"] = r.get("int8", {}).get("metrics", {}).get("f1")
            row["size_fp32_MB"] = r.get("fp32", {}).get("size_mb")
            row["size_int8_MB"] = r.get("int8", {}).get("size_mb")
        pc = (first(bench_pc, modelo=m, device="PC-CPU (onnx)", variante="int8")
              or first(bench_pc, modelo=m, device="PC-CPU (sklearn)"))
        row["lat_PC_CPU_ms"] = (pc or {}).get("lat_ms_mean")
        row["lat_Pi_CPU_ms"] = (first(bench_pi, modelo=m, backend="CPU onnx int8") or {}).get("lat_ms_mean")
        row["lat_Pi_NPU_ms"] = (first(bench_pi, modelo=m, backend="NPU (Hailo)") or {}).get("lat_ms_mean")
        pnpu = first(power_pi, modelo=m, backend="NPU (Hailo)")
        pcpu = first(power_pi, modelo=m, backend="CPU onnx int8")
        row["W_NPU_delta"] = (pnpu or {}).get("watts_delta")
        row["W_CPU_delta"] = (pcpu or {}).get("watts_delta")
        row["mJ_NPU"] = (pnpu or {}).get("mJ_por_infer")
        row["mJ_CPU"] = (pcpu or {}).get("mJ_por_infer")
        row["recall_-5dB_int8"] = recall_at(m, -5, "int8")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "comparativa_final.csv", index=False)

    # Tabla markdown
    cols = ["modelo", "familia", "params", "PR_AUC_fp32", "PR_AUC_int8",
            "size_int8_MB", "lat_PC_CPU_ms", "lat_Pi_CPU_ms", "lat_Pi_NPU_ms",
            "mJ_NPU", "mJ_CPU"]
    cols = [c for c in cols if c in df.columns]

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.4f}" if isinstance(v, float) and abs(v) < 100 else (
            f"{v:.1f}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df[cols].itertuples(index=False):
        md.append("| " + " | ".join(fmt(v) for v in r) + " |")
    (RES / "comparativa_final.md").write_text(
        "# Comparativa final: precisión, velocidad y consumo\n\n" + "\n".join(md) + "\n",
        encoding="utf-8")

    cnn = df[df["modelo"].isin(NPU)].copy()

    # Latencia en los 3 entornos (CNN)
    if not cnn.empty:
        sub = cnn.set_index("modelo")[["lat_PC_CPU_ms", "lat_Pi_CPU_ms", "lat_Pi_NPU_ms"]]
        sub = sub.reindex([m for m in ["cnn", "mobilenet", "cnn_big", "resnet"] if m in sub.index])
        ax = sub.plot(kind="bar", figsize=(9, 5), logy=True)
        ax.set_ylabel("latencia por ventana (ms, escala log)")
        ax.set_title("Latencia del mismo modelo en 3 entornos")
        ax.legend(["PC-CPU INT8", "Pi-CPU INT8", "Pi-NPU"])
        plt.xticks(rotation=0); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(FIG / "latencia_3entornos.png", dpi=150); plt.close()

    # Energía por inferencia NPU vs CPU (CNN)
    ecols = [c for c in ["mJ_NPU", "mJ_CPU"] if c in cnn.columns]
    if not cnn.empty and ecols:
        sub = cnn.set_index("modelo")[ecols]
        sub = sub.reindex([m for m in ["cnn", "mobilenet", "cnn_big", "resnet"] if m in sub.index])
        ax = sub.plot(kind="bar", figsize=(9, 5), logy=True,
                      color=["#2b6cb0", "#c05621"])
        ax.set_ylabel("energía por inferencia (mJ, escala log)")
        ax.set_title("Energía por inferencia: NPU vs CPU (Raspberry Pi)")
        ax.legend(["NPU", "CPU INT8"])
        plt.xticks(rotation=0); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(FIG / "energia_npu_vs_cpu.png", dpi=150); plt.close()

    # Pareto robustez vs energía (NPU)
    yfield = "recall_-5dB_int8" if cnn.get("recall_-5dB_int8") is not None else None
    if not cnn.empty and yfield and "mJ_NPU" in cnn.columns and cnn[yfield].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for _, r in cnn.iterrows():
            if r.get("mJ_NPU") is None or r.get(yfield) is None:
                continue
            ax.scatter(r["mJ_NPU"], r[yfield], s=110)
            ax.annotate(r["modelo"], (r["mJ_NPU"], r[yfield]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("energía por inferencia en NPU (mJ)  —  menos = mejor")
        ax.set_ylabel("robustez: recall de dron a -5 dB (INT8)")
        ax.set_title("Frontera de Pareto: robustez frente a coste energético (NPU)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "paretoez_energia.png", dpi=150); plt.close()

    print(df.to_string(index=False))
    print(f"\nGenerado: {RES}/comparativa_final.md, comparativa_final.csv")
    print(f"Figuras en {FIG}/: latencia_3entornos.png, energia_npu_vs_cpu.png, paretoez_energia.png")


SNR_GRID = [-10, -5, 0, 5, 10, 15, 20]


def _score(model, feat, device):
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0).unsqueeze(0).float().to(device)
        return torch.softmax(model(x), dim=1)[0, 1].item()


def robustez_main(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r")
    y = np.load(d / "labels.npy")
    te = np.load(d / "splits.npz")["test"]
    noise_pool = np.load(d / "noise_pool.npy")
    sc = np.load(d / "scaler.npz"); mean, std = sc["mean"], sc["std"]
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    sr, seg = cfg["audio"]["sample_rate"], cfg["audio"]["segment_seconds"]

    drone_idx = te[y[te] == 1]
    rng = np.random.default_rng(cfg["seed"])
    fig_dir = Path("results") / "figuras"; fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    summary = {}
    for name in args.models:
        ckpt_path = Path("checkpoints") / f"{name}_fp32.pth"
        if not ckpt_path.exists():
            print(f"[aviso] falta {ckpt_path}, se omite {name}"); continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(name, meta).to(device); model.load_state_dict(ckpt["state_dict"])
        model.eval()
        thr = ckpt.get("threshold", 0.5)

        recalls = []
        for snr in SNR_GRID:
            det = 0
            for i in drone_idx:
                drone = np.asarray(waves[i], dtype=np.float32)
                noise = noise_pool[rng.integers(len(noise_pool))].astype(np.float32)
                mixed = fix_length(mix_noise(drone, noise, snr), sr, seg)
                feat = (waveform_to_features(mixed, cfg) - mean) / std
                det += int(_score(model, feat, device) >= thr)
            recalls.append(det / len(drone_idx))
        summary[name] = recalls
        plt.plot(SNR_GRID, recalls, marker="o", label=name)
        print(f"{name:10s} recall@SNR " +
              " ".join(f"{s}:{r:.2f}" for s, r in zip(SNR_GRID, recalls)))

    plt.xlabel("SNR (dB)  —  menor = más ruido"); plt.ylabel("Recall de dron")
    plt.title("Robustez: detección de dron frente a ruido de fondo")
    plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / "robustez_snr.png", dpi=150); plt.close()
    with open(Path("results") / "robustez_snr.json", "w", encoding="utf-8") as fh:
        json.dump({"snr_grid": SNR_GRID, "recall": summary}, fh, indent=2)
    print(f"\nFigura: {fig_dir}/robustez_snr.png")




SNR_GRID = [-10, -5, 0, 5, 10, 15, 20]


def _softmax_row(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def robustez_int8_main(args):
    cfg = load_config(args.config)
    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r")
    y = np.load(d / "labels.npy")
    te = np.load(d / "splits.npz")["test"]
    noise_pool = np.load(d / "noise_pool.npy")
    sc = np.load(d / "scaler.npz"); mean, std = sc["mean"], sc["std"]
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    sr, seg = cfg["audio"]["sample_rate"], cfg["audio"]["segment_seconds"]
    drone_idx = te[y[te] == 1]
    fig_dir = Path("results") / "figuras"; fig_dir.mkdir(parents=True, exist_ok=True)

    # Precalcular las mezclas dron + ruido por SNR (mismas para FP32 e INT8)
    rng = np.random.default_rng(cfg["seed"])
    feats_by_snr = {}
    for snr in SNR_GRID:
        feats = []
        for i in drone_idx:
            drone = np.asarray(waves[i], dtype=np.float32)
            noise = noise_pool[rng.integers(len(noise_pool))].astype(np.float32)
            mixed = fix_length(mix_noise(drone, noise, snr), sr, seg)
            feats.append((waveform_to_features(mixed, cfg) - mean) / std)
        feats_by_snr[snr] = np.stack(feats)[:, None, :, :].astype(np.float32)

    summary = {}
    plt.figure(figsize=(9, 5.5))
    for model_name in args.models:
        pth = Path("checkpoints") / f"{model_name}_fp32.pth"
        onnx_int8 = Path("checkpoints") / f"{model_name}_int8.onnx"
        if not pth.exists():
            print(f"[aviso] falta {pth}, omito {model_name}"); continue
        ckpt = torch.load(pth, map_location="cpu")
        thr = ckpt.get("threshold", 0.5)
        model = build_model(model_name, meta); model.load_state_dict(ckpt["state_dict"]); model.eval()

        sess = None
        if onnx_int8.exists():
            sess = ort.InferenceSession(str(onnx_int8), providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name

        rec_fp32, rec_int8 = [], []
        for snr in SNR_GRID:
            X = feats_by_snr[snr]
            # Inferencia por lotes para no reservar tensores gigantes
            p_parts, pi_parts = [], []
            for s in range(0, len(X), args.batch):
                xb = X[s:s + args.batch]
                with torch.no_grad():
                    p_parts.append(torch.softmax(model(torch.from_numpy(xb)), dim=1)[:, 1].numpy())
                if sess is not None:
                    logits = sess.run(None, {in_name: xb})[0]
                    pi_parts.append(np.array([_softmax_row(r)[1] for r in logits]))
            p = np.concatenate(p_parts)
            rec_fp32.append(float((p >= thr).mean()))
            if sess is not None:
                pi = np.concatenate(pi_parts)
                rec_int8.append(float((pi >= thr).mean()))
        line, = plt.plot(SNR_GRID, rec_fp32, "o-", label=f"{model_name} FP32")
        summary[model_name] = {"fp32": rec_fp32}
        if sess is not None:
            plt.plot(SNR_GRID, rec_int8, "x--", color=line.get_color(), label=f"{model_name} INT8")
            summary[model_name]["int8"] = rec_int8
        print(f"{model_name}: FP32 {['%.2f'%r for r in rec_fp32]}"
              + (f" | INT8 {['%.2f'%r for r in rec_int8]}" if sess is not None else ""))

    plt.xlabel("SNR (dB)  —  menor = más ruido"); plt.ylabel("Recall de dron")
    plt.title("Robustez frente a ruido: FP32 vs INT8")
    plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(fig_dir / "robustez_snr_fp32_int8.png", dpi=150); plt.close()
    with open(Path("results") / "robustez_snr_quant.json", "w", encoding="utf-8") as fh:
        json.dump({"snr_grid": SNR_GRID, "recall": summary}, fh, indent=2)
    print(f"\nFigura: {fig_dir}/robustez_snr_fp32_int8.png")


NEURAL = ["cnn", "cnn_big", "resnet", "mobilenet", "crnn", "lstm", "gru"]
CLASSICAL = ["svm", "random_forest", "knn"]


def _timeit(fn, x, warmup, reps):
    for _ in range(warmup):
        fn(x)
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter(); fn(x); ts[i] = (time.perf_counter() - t0) * 1000.0
    return {"lat_ms_mean": float(ts.mean()), "lat_ms_std": float(ts.std()),
            "lat_ms_p95": float(np.percentile(ts, 95))}


def bench_pc_main(args):
    cfg = load_config(args.config)
    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r")
    te = np.load(d / "splits.npz")["test"]
    sc = np.load(d / "scaler.npz"); mean, std = sc["mean"], sc["std"]
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    idx = te[:args.n]
    print(f"Preparando {len(idx)} muestras de test para el benchmark...")
    raw_feats = [waveform_to_features(np.asarray(waves[i], dtype=np.float32), cfg) for i in idx]
    Ximg = np.stack([(f - mean) / std for f in raw_feats])[:, None, :, :].astype(np.float32)
    x1 = Ximg[:1]
    gpu = torch.cuda.is_available()
    ckpt = Path("checkpoints"); rows = []

    for m in args.models:
        pth = ckpt / f"{m}_fp32.pth"
        if not pth.exists():
            print(f"[aviso] falta {pth}, se omite {m}"); continue
        c = torch.load(pth, map_location="cpu")
        model = build_model(m, meta); model.load_state_dict(c["state_dict"]); model.eval()
        nparams = sum(p.numel() for p in model.parameters())

        # FP32 CPU (torch), lote 1
        xt = torch.from_numpy(x1)
        r = _timeit(lambda z: model(z), xt, args.warmup, args.reps)
        # Throughput por lotes (CPU)
        xb = torch.from_numpy(Ximg)
        with torch.no_grad():
            t0 = time.perf_counter(); model(xb); thr_cpu = len(Ximg) / (time.perf_counter() - t0)
        rows.append({"modelo": m, "familia": "CNN" if m in ["cnn", "cnn_big", "resnet", "mobilenet"] else "recurrente",
                     "variante": "fp32", "device": "PC-CPU (torch)", "params": nparams,
                     "throughput_lote_sps": round(thr_cpu, 1), **r})

        # FP32 GPU (torch), lote 1
        if gpu:
            mg = model.to("cuda"); xg = xt.to("cuda")
            def gfn(z):
                with torch.no_grad():
                    mg(z); torch.cuda.synchronize()
            rg = _timeit(gfn, xg, args.warmup, args.reps)
            rows.append({"modelo": m, "familia": rows[-1]["familia"], "variante": "fp32",
                         "device": "PC-GPU (torch)", "params": nparams,
                         "throughput_lote_sps": None, **rg})
            model.to("cpu")

        # INT8 CPU (onnx), lote 1
        onnx = ckpt / f"{m}_int8.onnx"
        if onnx.exists():
            sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            nm = sess.get_inputs()[0].name
            ri = _timeit(lambda z: sess.run(None, {nm: z}), x1, args.warmup, args.reps)
            t0 = time.perf_counter(); sess.run(None, {nm: Ximg}); thr_i = len(Ximg) / (time.perf_counter() - t0)
            rows.append({"modelo": m, "familia": rows[-1]["familia"] if not gpu else
                         ("CNN" if m in ["cnn", "cnn_big", "resnet", "mobilenet"] else "recurrente"),
                         "variante": "int8", "device": "PC-CPU (onnx)", "params": nparams,
                         "throughput_lote_sps": round(thr_i, 1), **ri})
        print(f"  {m}: medido")

    # Clásicos
    fsc = ckpt / "classical_feature_scaler.pkl"
    if fsc.exists() and any((ckpt / f"classical_{n}.pkl").exists() for n in CLASSICAL):
        Xstat = mfcc_statistics(np.stack(raw_feats))
        scaler = joblib.load(fsc); Xs = scaler.transform(Xstat).astype(np.float32); xs1 = Xs[:1]
        for name in CLASSICAL:
            p = ckpt / f"classical_{name}.pkl"
            if not p.exists():
                continue
            clf = joblib.load(p)["model"]
            r = _timeit(lambda z: clf.predict_proba(z), xs1, min(args.warmup, 10), min(args.reps, 100))
            t0 = time.perf_counter(); clf.predict_proba(Xs); thr_c = len(Xs) / (time.perf_counter() - t0)
            rows.append({"modelo": name, "familia": "clásico", "variante": "fp32",
                         "device": "PC-CPU (sklearn)", "params": None,
                         "throughput_lote_sps": round(thr_c, 1), **r})
            print(f"  {name}: medido")

    import pandas as pd
    df = pd.DataFrame(rows)
    res = Path("results"); res.mkdir(exist_ok=True)
    df.to_csv(res / "bench_pc.csv", index=False)
    with open(res / "bench_pc.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    # tabla markdown
    cols = ["modelo", "familia", "variante", "device", "lat_ms_mean", "lat_ms_std",
            "lat_ms_p95", "throughput_lote_sps"]
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df[cols].itertuples(index=False):
        md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))
                                     for v in r) + " |")
    (res / "bench_pc.md").write_text("# Latencia de inferencia en PC (lote 1)\n\n" + "\n".join(md) + "\n",
                                     encoding="utf-8")

    # Latencia CPU lote 1
    fig_dir = res / "figuras"; fig_dir.mkdir(parents=True, exist_ok=True)
    cpu = df[df["device"].isin(["PC-CPU (torch)", "PC-CPU (onnx)", "PC-CPU (sklearn)"])]
    if not cpu.empty:
        piv = cpu.pivot_table(index="modelo", columns="variante", values="lat_ms_mean")
        ax = piv.plot(kind="bar", figsize=(10, 4.5))
        ax.set_ylabel("latencia por ventana (ms), lote 1"); ax.set_title("Inferencia en CPU del PC")
        plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(fig_dir / "latencia_pc_cpu.png", dpi=150); plt.close()

    print(df.to_string(index=False))
    print(f"\nGuardado en {res}/bench_pc.md, bench_pc.csv y {fig_dir}/latencia_pc_cpu.png")


MODELS = ["cnn", "cnn_big", "resnet", "mobilenet"]


def _timeit_pi(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter(); fn(); ts[i] = (time.perf_counter() - t0) * 1000.0
    return {"lat_ms_mean": float(ts.mean()), "lat_ms_std": float(ts.std()),
            "lat_ms_p95": float(np.percentile(ts, 95)),
            "throughput_sps": float(1000.0 / ts.mean())}


def bench_hailo(hef_path, warmup, reps):
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
    )
    hef = HEF(hef_path)
    vdev = VDevice()
    ng = vdev.configure(hef, ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe))[0]
    ngp = ng.create_params()
    inp = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    outp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    x = np.random.randn(1, *tuple(in_info.shape)).astype(np.float32)
    try:
        with InferVStreams(ng, inp, outp) as pipe:
            with ng.activate(ngp):
                r = _timeit_pi(lambda: pipe.infer({in_info.name: x}), warmup, reps)
    finally:
        try:
            vdev.release()
        except Exception:
            pass
    return r


def bench_onnx(onnx_path, warmup, reps, F, T):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    x = np.random.randn(1, 1, F, T).astype(np.float32)
    return _timeit_pi(lambda: sess.run(None, {nm: x}), warmup, reps)


def bench_pi_main(args):
    with open("data/processed/meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    F, T = meta["n_features"], meta["n_frames"]

    rows = []
    for m in args.models:
        hef = Path("checkpoints") / f"{m}_hailo8.hef"
        if hef.exists():
            try:
                r = bench_hailo(str(hef), args.warmup, args.reps)
                rows.append({"modelo": m, "backend": "NPU (Hailo)", "device": "Pi-NPU", **r})
                print(f"  {m:10s} NPU        {r['lat_ms_mean']:.3f} ± {r['lat_ms_std']:.3f} ms")
            except Exception as e:
                print(f"  {m}: NPU falló: {e}")
        for prec in ["int8", "fp32"]:
            onnx = Path("checkpoints") / f"{m}_{prec}.onnx"
            if onnx.exists():
                r = bench_onnx(str(onnx), args.warmup, args.reps, F, T)
                rows.append({"modelo": m, "backend": f"CPU onnx {prec}", "device": "Pi-CPU", **r})
                print(f"  {m:10s} CPU-{prec:4s}  {r['lat_ms_mean']:.3f} ± {r['lat_ms_std']:.3f} ms")

    res = Path("results"); res.mkdir(exist_ok=True)
    with open(res / "bench_pi.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    cols = ["modelo", "backend", "device", "lat_ms_mean", "lat_ms_std", "lat_ms_p95", "throughput_sps"]
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (res / "bench_pi.md").write_text(
        "# Latencia de inferencia en la Raspberry Pi (lote 1)\n\n" + "\n".join(md) + "\n",
        encoding="utf-8")
    print(f"\nGuardado en {res}/bench_pi.md y bench_pi.json")


MODELS = ["cnn", "cnn_big", "resnet", "mobilenet"]


def read_power_w():
    out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True)
    amps, volts = {}, {}
    for name, kind, val in re.findall(r"(\S+)_([AV])\s+\w+\(\d+\)=([0-9.]+)[AV]", out):
        (amps if kind == "A" else volts)[name] = float(val)
    return sum(a * volts.get(k, 0.0) for k, a in amps.items())


def _sampler(stop_evt, samples, period=0.1):
    while not stop_evt.is_set():
        try:
            samples.append(read_power_w())
        except Exception:
            pass
        time.sleep(period)


def measure(loop_fn, seconds):
    stop = threading.Event(); samples = []
    th = threading.Thread(target=_sampler, args=(stop, samples)); th.start()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < seconds:
        loop_fn(); n += 1
    dur = time.perf_counter() - t0
    stop.set(); th.join()
    w = np.array(samples) if samples else np.array([float("nan")])
    thr = n / dur
    return {"watts_mean": float(w.mean()), "watts_std": float(w.std()),
            "throughput_sps": thr, "n": n, "dur_s": dur}


def build_onnx(onnx_path, F, T):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    x = np.random.randn(1, 1, F, T).astype(np.float32)
    return (lambda: sess.run(None, {nm: x})), (lambda: None)


def build_hailo(hef_path):
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
    )
    hef = HEF(hef_path); vdev = VDevice()
    ng = vdev.configure(hef, ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe))[0]
    ngp = ng.create_params()
    inp = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    outp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    in_info = hef.get_input_vstream_infos()[0]
    x = np.random.randn(1, *tuple(in_info.shape)).astype(np.float32)
    pipe = InferVStreams(ng, inp, outp); pipe.__enter__()
    act = ng.activate(ngp); act.__enter__()

    def infer():
        pipe.infer({in_info.name: x})

    def cleanup():
        try: act.__exit__(None, None, None)
        except Exception: pass
        try: pipe.__exit__(None, None, None)
        except Exception: pass
        try: vdev.release()
        except Exception: pass
    return infer, cleanup


def consumo_main(args):
    with open("data/processed/meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    F, T = meta["n_features"], meta["n_frames"]

    print(f"Midiendo reposo {args.secs}s...")
    idle = measure(lambda: time.sleep(0.02), args.secs)
    print(f"  Reposo: {idle['watts_mean']:.2f} ± {idle['watts_std']:.2f} W\n")

    rows = [{"modelo": "—", "backend": "reposo", **idle, "watts_delta": 0.0, "mJ_por_infer": None}]
    for m in args.models:
        for backend, builder in [
            ("NPU (Hailo)", lambda mm=m: build_hailo(f"checkpoints/{mm}_hailo8.hef")),
            ("CPU onnx int8", lambda mm=m: build_onnx(f"checkpoints/{mm}_int8.onnx", F, T)),
        ]:
            path_ok = (Path("checkpoints") / (f"{m}_hailo8.hef" if "NPU" in backend
                       else f"{m}_int8.onnx")).exists()
            if not path_ok:
                continue
            try:
                infer, cleanup = builder()
            except Exception as e:
                print(f"  {m} {backend}: no disponible ({e})"); continue
            r = measure(infer, args.secs)
            cleanup()
            delta = r["watts_mean"] - idle["watts_mean"]
            mj = 1000.0 * r["watts_mean"] / r["throughput_sps"] # W / (inf/s) = J/inf -> mJ
            rows.append({"modelo": m, "backend": backend, **r,
                         "watts_delta": delta, "mJ_por_infer": mj})
            print(f"  {m:10s} {backend:14s} {r['watts_mean']:.2f} W "
                  f"(+{delta:.2f}) | {r['throughput_sps']:.0f} inf/s | {mj:.1f} mJ/infer")

    res = Path("results"); res.mkdir(exist_ok=True)
    with open(res / "power_pi.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {res}/power_pi.json")


def _cli():
    ap = argparse.ArgumentParser(description="Análisis de resultados")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tabla", help="tabla de precisión de todos los modelos")
    sub.add_parser("cuantizacion", help="tabla y figuras de cuantización")
    sub.add_parser("final", help="comparativa final + figura de Pareto")

    for nombre in ("robustez", "robustez-int8"):
        s = sub.add_parser(nombre, help="curvas de robustez frente a ruido")
        s.add_argument("--config", default="config.yaml")
        s.add_argument("--models", nargs="+",
                       default=["cnn", "cnn_big", "resnet", "mobilenet", "crnn", "lstm", "gru"])
        if nombre == "robustez-int8":
            s.add_argument("--batch", type=int, default=256)

    b = sub.add_parser("bench-pc", help="latencia de inferencia en el PC")
    b.add_argument("--config", default="config.yaml")
    b.add_argument("--models", nargs="+", default=NEURAL)
    b.add_argument("--n", type=int, default=256)
    b.add_argument("--warmup", type=int, default=20)
    b.add_argument("--reps", type=int, default=200)

    bp = sub.add_parser("bench-pi", help="latencia NPU vs CPU en la Raspberry Pi")
    bp.add_argument("--models", nargs="+", default=MODELS)
    bp.add_argument("--warmup", type=int, default=20)
    bp.add_argument("--reps", type=int, default=200)

    c = sub.add_parser("consumo", help="consumo por PMIC en la Raspberry Pi")
    c.add_argument("--models", nargs="+", default=MODELS)
    c.add_argument("--secs", type=float, default=12.0)

    args = ap.parse_args()
    if args.cmd == "tabla":
        tabla_modelos_main()
    elif args.cmd == "cuantizacion":
        tabla_cuant_main()
    elif args.cmd == "final":
        comparativa_final_main()
    elif args.cmd == "robustez":
        robustez_main(args)
    elif args.cmd == "robustez-int8":
        robustez_int8_main(args)
    elif args.cmd == "bench-pc":
        bench_pc_main(args)
    elif args.cmd == "bench-pi":
        bench_pi_main(args)
    elif args.cmd == "consumo":
        consumo_main(args)


if __name__ == "__main__":
    _cli()
