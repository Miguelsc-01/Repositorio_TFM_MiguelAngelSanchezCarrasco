# Entrenamiento (redes y clásicos), cuantización INT8 y compilación a HEF.

# Comandos:
    # python entrenar.py red --model resnet --epochs 30 --patience 6 --batch-size 128 --workers 6
    # python entrenar.py clasicos
    # python entrenar.py cuantizar
    # python entrenar.py calibracion

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat, CalibrationDataReader,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utiles import (
    load_config, set_seed, classification_metrics, metrics_at_threshold,
    best_f1_threshold, format_metrics, save_confusion_matrix,
)
from modelos import build_model, build_classifiers, AVAILABLE_TORCH_MODELS
from datos import RobustDroneDataset, waveform_to_features, mfcc_statistics


# Limitar hilos BLAS a 1 antes de importar numpy, ya que con varios workers, cada uno
# lanzaría sus propios hilos OpenBLAS y agotaría la memoria.
# Los procesos hijos heredan estas variables.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")




@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, scores = [], [], []
    for xb, yb in loader:
        prob = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
        preds.append((prob >= 0.5).long().cpu().numpy())
        scores.append(prob.cpu().numpy())
        ys.append(np.asarray(yb))
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(scores)


def entrenar_red(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r") # No duplica en workers
    labels = np.load(d / "labels.npy")
    sp = np.load(d / "splits.npz")
    sc = np.load(d / "scaler.npz")
    noise_pool = np.load(d / "noise_pool.npy")
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    mean, std = sc["mean"], sc["std"]
    ds_tr = RobustDroneDataset(waves, labels, cfg, noise_pool, mean, std, train=True)
    ds_va = RobustDroneDataset(waves, labels, cfg, None, mean, std, train=False)
    ds_te = RobustDroneDataset(waves, labels, cfg, None, mean, std, train=False)
    # subconjuntos por índices
    from torch.utils.data import Subset
    tr_kw = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        tr_kw.update(persistent_workers=True, prefetch_factor=2)    # Evita re-spawn por época
    # val/test: pocos workers y no persistentes (evita apilar procesos que importan CUDA y agotan el archivo de paginación en Windows)
    ev_workers = min(2, args.workers)
    ev_kw = dict(num_workers=ev_workers, pin_memory=True)
    if ev_workers > 0:
        ev_kw.update(prefetch_factor=2)
    ld_tr = DataLoader(Subset(ds_tr, tr), batch_size=args.batch_size, shuffle=True,
                       drop_last=False, **tr_kw)
    ld_va = DataLoader(Subset(ds_va, va), batch_size=args.batch_size, **ev_kw)
    ld_te = DataLoader(Subset(ds_te, te), batch_size=args.batch_size, **ev_kw)

    model = build_model(args.model, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo '{args.model}' con {n_params:,} parámetros | "
          f"entrada (1, {meta['n_features']}, {meta['n_frames']})")

    cw = meta["class_weight"]
    weight = torch.tensor([cw.get("0", cw.get(0)), cw.get("1", cw.get(1))],
                          dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    res_dir = Path("results"); res_dir.mkdir(exist_ok=True)
    best_path = ckpt_dir / f"{args.model}_fp32.pth"

    best_pr, no_improve, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in ld_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()
        yt, _, st = evaluate(model, ld_va, device)
        vm = classification_metrics(yt, (st >= 0.5).astype(int), st)
        history.append({"epoch": epoch, "loss": running / len(tr), "val": vm})
        print(f"Epoch {epoch:03d} | loss {running/len(tr):.4f} | "
              f"val PR-AUC {vm['pr_auc']:.4f} F1 {vm['f1']:.4f} bal {vm['balanced_accuracy']:.4f}")
        if vm["pr_auc"] > best_pr:
            best_pr, no_improve = vm["pr_auc"], 0
            torch.save({"state_dict": model.state_dict(), "model": args.model, "meta": meta}, best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping en epoch {epoch}.")
                break

    # Liberar los workers de train antes de la evaluación final (libera memoria)
    del ld_tr
    import gc; gc.collect()

    if not best_path.exists():
        print(f"\n[ERROR] El modelo '{args.model}' no llegó a converger (loss/val nan), no se guardó ningún checkpoint. Prueba con un learning rate menor, p.ej. --lr 5e-4.")
        return

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    yv, _, sv = evaluate(model, ld_va, device)
    thr = best_f1_threshold(yv, sv)
    yt, _, st = evaluate(model, ld_te, device)
    tm05 = metrics_at_threshold(yt, st, 0.5)
    tmthr = metrics_at_threshold(yt, st, thr)
    print(f"\n=== TEST ({args.model}, val PR-AUC={best_pr:.4f}) ===")
    print("[umbral 0.50]"); print(format_metrics(tm05))
    print(f"[umbral {thr:.3f}]"); print(format_metrics(tmthr))

    ckpt["threshold"] = thr
    torch.save(ckpt, best_path)
    np.savez(res_dir / f"scores_{args.model}.npz", y_true=yt, y_score=st)
    save_confusion_matrix(tmthr["confusion_matrix"], res_dir / f"cm_{args.model}.png",
                          title=f"{args.model}")
    with open(res_dir / f"results_{args.model}.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": thr, "test_thr05": tm05, "test_tuned": tmthr,
                   "best_val_pr_auc": best_pr, "n_params": n_params, "history": history},
                  fh, indent=2, ensure_ascii=False)
    print(f"\nCheckpoint: {best_path}")


def entrenar_clasicos(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r")
    y = np.load(d / "labels.npy")
    sp = np.load(d / "splits.npz")
    tr, va, te = sp["train"], sp["val"], sp["test"]

    # log-mel -> estadísticos por banda (sin aumentos)
    feats = [waveform_to_features(np.asarray(waves[i], dtype=np.float32), cfg)
             for i in tqdm(range(len(y)), desc="Log-mel")]
    S = mfcc_statistics(np.stack(feats))            # (N, n_mels*5)
    scaler = StandardScaler().fit(S[tr])
    S = scaler.transform(S).astype(np.float32)

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    res_dir = Path("results"); res_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, ckpt_dir / "classical_feature_scaler.pkl")

    all_results = {}
    for name, clf in build_classifiers(cfg["seed"]).items():
        print(f"\n=== {name.upper()} ===")
        clf.fit(S[tr], y[tr])
        sv, st = clf.predict_proba(S[va])[:, 1], clf.predict_proba(S[te])[:, 1]
        thr = best_f1_threshold(y[va], sv)
        test_05 = metrics_at_threshold(y[te], st, 0.5)
        test_thr = metrics_at_threshold(y[te], st, thr)
        print(f"[test umbral {thr:.3f}]"); print(format_metrics(test_thr))

        joblib.dump({"model": clf, "threshold": thr}, ckpt_dir / f"classical_{name}.pkl")
        save_confusion_matrix(test_thr["confusion_matrix"],
                              res_dir / f"cm_{name}.png", title=f"{name}")
        np.savez(res_dir / f"scores_{name}.npz", y_true=y[te], y_score=st)
        # Guardado con el mismo formato que los modelos profundos
        with open(res_dir / f"results_{name}.json", "w", encoding="utf-8") as fh:
            json.dump({"threshold": thr, "test_thr05": test_05, "test_tuned": test_thr,
                       "n_params": None}, fh, indent=2, ensure_ascii=False)
        all_results[name] = test_thr

    print("\n=== Resumen clásicos (test) por PR-AUC ===")
    for name, m in sorted(all_results.items(), key=lambda kv: kv[1]["pr_auc"], reverse=True):
        print(f"  {name:14s} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} bal={m['balanced_accuracy']:.4f}")


CNN_MODELS = ["cnn", "cnn_big", "resnet", "mobilenet"]  # ejecutables en NPU
RNN_MODELS = ["crnn", "lstm", "gru"]    # solo CPU (Hailo no soporta RNN)
NEURAL_MODELS = CNN_MODELS + RNN_MODELS
NPU_CAPABLE = set(CNN_MODELS)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def features_for(waves, idx, cfg, mean, std):
    # Log-mel escalado (sin aumentos) para un conjunto de índices -> (N,1,F,T)
    feats = [(waveform_to_features(np.asarray(waves[i], dtype=np.float32), cfg) - mean) / std
             for i in idx]
    return np.stack(feats)[:, None, :, :].astype(np.float32)


def export_onnx(model_name, ckpt_path, onnx_path, meta):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(model_name, meta)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.randn(1, 1, meta["n_features"], meta["n_frames"])
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return float(ckpt.get("threshold", 0.5))


def to_fp16(onnx_fp32, onnx_fp16):
    from onnxconverter_common import float16
    m = onnx.load(onnx_fp32)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, onnx_fp16)


class _CalibReader(CalibrationDataReader):
    def __init__(self, X, input_name, limit=300):
        self.samples = [{input_name: X[i:i + 1].astype(np.float32)}
                        for i in range(min(limit, len(X)))]
        self._it = iter(self.samples)

    def get_next(self):
        return next(self._it, None)


def to_int8(onnx_fp32, onnx_int8, X_calib):
    pre = str(onnx_fp32).replace(".onnx", "_pre.onnx")
    quant_pre_process(onnx_fp32, pre)
    sess = ort.InferenceSession(pre, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = _CalibReader(X_calib, input_name)
    quantize_static(
        pre, onnx_int8, reader,
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    )
    Path(pre).unlink(missing_ok=True)


def eval_onnx(onnx_path, X_test, y_test, threshold, lat_samples=500, batch=256):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    probs = []
    for s in range(0, len(X_test), batch):  # Por lotes: evita reservar tensores enormes
        logits = sess.run(None, {name: X_test[s:s + batch].astype(np.float32)})[0]
        probs.append(_softmax(logits)[:, 1])
    prob = np.concatenate(probs)
    m = metrics_at_threshold(y_test, prob, threshold)
    n = min(lat_samples, len(X_test))
    _ = sess.run(None, {name: X_test[0:1].astype(np.float32)})
    t0 = time.perf_counter()
    for i in range(n):
        sess.run(None, {name: X_test[i:i + 1].astype(np.float32)})
    latency_ms = (time.perf_counter() - t0) / n * 1000.0
    size_mb = Path(onnx_path).stat().st_size / 1e6
    return m, latency_ms, size_mb, prob


def quantize_model(model_name, X_test, y_test, X_calib, meta, ckpt_dir, res_dir):
    pth = ckpt_dir / f"{model_name}_fp32.pth"
    if not pth.exists():
        print(f"[aviso] no existe {pth}, se omite {model_name}")
        return None
    fp32 = ckpt_dir / f"{model_name}_fp32.onnx"
    fp16 = ckpt_dir / f"{model_name}_fp16.onnx"
    int8 = ckpt_dir / f"{model_name}_int8.onnx"

    print(f"\n=== {model_name} ===")
    thr = export_onnx(model_name, pth, str(fp32), meta)
    print(f"  umbral de operación: {thr:.3f}")
    to_fp16(str(fp32), str(fp16))
    try:
        to_int8(str(fp32), str(int8), X_calib)
        int8_ok = True
    except Exception as e:
        print(f"  INT8 no completado: {e}")
        int8_ok = False

    variants = [("fp32", fp32), ("fp16", fp16)]
    if int8_ok:
        variants.append(("int8", int8))

    results = {}
    print(f"  {'variante':8s} {'PR-AUC':>8s} {'F1':>7s} {'tam(MB)':>8s} {'lat(ms)':>8s}")
    for tag, path in variants:
        m, lat, size, prob = eval_onnx(str(path), X_test, y_test, thr)
        results[tag] = {"metrics": m, "latency_ms": lat, "size_mb": size}
        np.savez(res_dir / f"scores_{model_name}_{tag}.npz",
                 y_true=y_test, y_score=prob)
        print(f"  {tag:8s} {m['pr_auc']:8.4f} {m['f1']:7.4f} {size:8.3f} {lat:8.3f}")

    base = results["fp32"]
    for tag in [t for t, _ in variants if t != "fp32"]:
        d_pr = results[tag]["metrics"]["pr_auc"] - base["metrics"]["pr_auc"]
        red = 100 * (1 - results[tag]["size_mb"] / base["size_mb"])
        spd = base["latency_ms"] / results[tag]["latency_ms"]
        print(f"    {tag}: dPR-AUC={d_pr:+.4f}  tamaño -{red:.1f}%  speedup CPU x{spd:.2f}")

    n_params = sum(v.numel() for v in build_model(model_name, meta).parameters())
    npu = model_name in NPU_CAPABLE
    out = {"model": model_name, "threshold": thr, "n_params": n_params,
           "npu_capable": npu, "family": "CNN" if npu else "recurrente",
           "device": "NPU" if npu else "CPU", "results": results}
    with open(res_dir / f"quantization_{model_name}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    return out


def cuantizar(args):
    cfg = load_config(args.config)
    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r")
    y = np.load(d / "labels.npy")
    sp = np.load(d / "splits.npz"); tr, te = sp["train"], sp["test"]
    sc = np.load(d / "scaler.npz"); mean, std = sc["mean"], sc["std"]
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    print("Extrayendo log-mel de test y calibración (una vez)...")
    X_test = features_for(waves, te, cfg, mean, std)
    calib_idx = tr[:args.calib]
    X_calib = features_for(waves, calib_idx, cfg, mean, std)

    ckpt_dir = Path("checkpoints")
    res_dir = Path("results"); res_dir.mkdir(exist_ok=True)
    for model_name in args.models:
        quantize_model(model_name, X_test, y[te], X_calib, meta, ckpt_dir, res_dir)
    print(f"\nHecho. JSONs de cuantización en {res_dir.resolve()}")


def hacer_calibracion(args):
    cfg = load_config(args.config)
    d = Path(cfg["dataset"]["processed_dir"])
    waves = np.load(d / "waves.npy", mmap_mode="r")
    tr = np.load(d / "splits.npz")["train"]
    sc = np.load(d / "scaler.npz"); mean, std = sc["mean"], sc["std"]

    idx = tr[:args.calib]
    print(f"Calculando log-mel de {len(idx)} muestras de train para calibración...")
    feats = [(waveform_to_features(np.asarray(waves[i], dtype=np.float32), cfg) - mean) / std
             for i in idx]
    calib = np.stack(feats)[:, :, :, None].astype(np.float32)   # (N, F, T, 1) NHWC

    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    path = out / "calib_nhwc.npy"
    np.save(path, calib)
    print(f"Guardado {path}  shape {calib.shape}  "
          f"(rango [{calib.min():.2f}, {calib.max():.2f}])")
    print("Copiar a WSL junto con los ONNX para compilar los HEF (compilar.py).")

    print("Copiar a la Raspberry Pi junto con data/processed/scaler.npz, "
          "meta.json, configs/ y results/results_" + args.model + ".json (umbral).")


def _cli():
    ap = argparse.ArgumentParser(description="Entrenamiento, cuantización y compilación")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("red", help="entrena una red (cnn, resnet, lstm, ...)")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--model", default="resnet",
                   choices=["cnn", "cnn_big", "resnet", "mobilenet", "crnn", "lstm", "gru"])
    r.add_argument("--epochs", type=int, default=60)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--patience", type=int, default=12)
    r.add_argument("--workers", type=int, default=4)

    c = sub.add_parser("clasicos", help="entrena SVM, Random Forest y KNN")
    c.add_argument("--config", default="config.yaml")

    q = sub.add_parser("cuantizar", help="genera FP32/FP16/INT8 de las redes")
    q.add_argument("--config", default="config.yaml")
    q.add_argument("--models", nargs="+", default=NEURAL_MODELS, choices=NEURAL_MODELS)
    q.add_argument("--calib", type=int, default=300)

    k = sub.add_parser("calibracion", help="precalcula datos de calibración para Hailo")
    k.add_argument("--config", default="config.yaml")
    k.add_argument("--calib", type=int, default=1024)

    args = ap.parse_args()
    if args.cmd == "red":
        entrenar_red(args)
    elif args.cmd == "clasicos":
        entrenar_clasicos(args.config)
    elif args.cmd == "cuantizar":
        cuantizar(args)
    elif args.cmd == "calibracion":
        hacer_calibracion(args)


if __name__ == "__main__":
    _cli()
