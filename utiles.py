# Utilidades compartidas: configuración, semilla y métricas
from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    average_precision_score, roc_auc_score, confusion_matrix,
)


def load_config(path: str | Path) -> dict:
    # Lee un fichero YAML de configuración y lo devuelve como diccionario
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def set_seed(seed: int = 42) -> None:
    # Fija las semillas de random, numpy y si está disponible torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    # Calcula el diccionario de métricas. `y_score` = prob. de clase dron
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            m["pr_auc"] = float(average_precision_score(y_true, y_score))
            m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            m["pr_auc"] = float("nan")
            m["roc_auc"] = float("nan")
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return m


def metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    # Métricas aplicando un umbral concreto a las puntuaciones
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return classification_metrics(y_true, y_pred, y_score)


def best_f1_threshold(y_true, y_score) -> float:
    # Recorre la curva precision-recall y escoge el punto de mayor F1, por lo que devuelve el umbral que maximiza el F1, se elige solo en validación
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.nanargmax(f1[:-1]))   # thr tiene un elemento menos que prec/rec
    return float(thr[idx])


def format_metrics(m: dict) -> str:
    # Formatea las métricas para imprimir en consola
    cm = m["confusion_matrix"]
    lines = [
        f"  accuracy           : {m['accuracy']:.4f}",
        f"  balanced_accuracy  : {m['balanced_accuracy']:.4f}",
        f"  precision (dron)   : {m['precision']:.4f}",
        f"  recall (dron)      : {m['recall']:.4f}",
        f"  f1 (dron)          : {m['f1']:.4f}",
    ]
    if "pr_auc" in m:
        lines.append(f"  PR-AUC             : {m['pr_auc']:.4f}")
        lines.append(f"  ROC-AUC            : {m['roc_auc']:.4f}")
    lines.append(f"  confusion [[TN,FP],[FN,TP]] : {cm}")
    return "\n".join(lines)


def save_confusion_matrix(cm, path, title="Matriz de confusión") -> None:
    # Guarda la matriz de confusión como imagen
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["no-dron", "dron"],
                yticklabels=["no-dron", "dron"], ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
