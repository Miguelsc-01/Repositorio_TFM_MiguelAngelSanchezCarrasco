# Descarga de datasets, preprocesado, features y aumentos.

# Comandos:
    # python datos.py preparar --config config.yaml
    # python datos.py descargar --dataset ... --out ... [--window 1.0 --hop 1.0 --max N]

from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utiles import load_config, set_seed

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def load_audio(path: str, sample_rate: int, segment_seconds: float,
               mono: bool = True) -> np.ndarray:
    # Carga un fichero de audio y lo ajusta a una longitud fija.
    # Si el clip es más corto que segment_seconds se rellena con ceros y
    # si es más largo, se recorta. Devuelve un vector de longitud sample_rate * segment_seconds

    y, _ = librosa.load(path, sr=sample_rate, mono=mono)
    return fix_length(y, sample_rate, segment_seconds)


def fix_length(y: np.ndarray, sample_rate: int, segment_seconds: float) -> np.ndarray:
    # Rellena o recorta la señal a una longitud fija en muestras
    target = int(round(sample_rate * segment_seconds))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y.astype(np.float32)


def waveform_to_features(y: np.ndarray, cfg: dict) -> np.ndarray:
    # Convierte una forma de onda de longitud fija en un mapa de características de forma (n_features, n_frames).
    # Trabaja directamente sobre el vector de audio para poder reutilizarse con el micrófono en tiempo real sin pasar por disco.

    f = cfg["features"]
    sr = cfg["audio"]["sample_rate"]

    if f["type"] == "mfcc":
        feat = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=f["n_mfcc"], n_fft=f["n_fft"], hop_length=f["hop_length"],
            n_mels=f["n_mels"], fmin=f["fmin"], fmax=f["fmax"],
        )
    elif f["type"] == "logmel":
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=f["n_fft"], hop_length=f["hop_length"],
            n_mels=f["n_mels"], fmin=f["fmin"], fmax=f["fmax"],
        )
        feat = librosa.power_to_db(mel, ref=np.max)
    else:
        raise ValueError(f"features.type desconocido: {f['type']}")

    if f.get("deltas", False):
        d1 = librosa.feature.delta(feat)
        d2 = librosa.feature.delta(feat, order=2)
        feat = np.concatenate([feat, d1, d2], axis=0)

    return feat.astype(np.float32)


def extract_from_file(path: str, cfg: dict) -> np.ndarray:
    # Carga un fichero y devuelve su mapa de características
    y = load_audio(path, cfg["audio"]["sample_rate"],
                   cfg["audio"]["segment_seconds"], cfg["audio"]["mono"])
    return waveform_to_features(y, cfg)


def mix_noise(drone: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # Mezcla drone + noise escalando el ruido al SNR pedido (en dB)
    drone = drone.astype(np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < len(drone): # repetir si es corto
        reps = int(np.ceil(len(drone) / max(1, len(noise))))
        noise = np.tile(noise, reps)
    noise = noise[:len(drone)]

    p_drone = float(np.mean(drone ** 2)) + 1e-10
    p_noise = float(np.mean(noise ** 2)) + 1e-10
    target_noise_power = p_drone / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / p_noise)

    mixed = drone + noise
    peak = float(np.max(np.abs(mixed))) + 1e-9  # evita saturación
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    # Desplaza el tono n_steps semitonos para cubrir drones más agudos/graves
    return librosa.effects.pitch_shift(y=y.astype(np.float32), sr=sr,
                                       n_steps=float(n_steps)).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    # Estira/comprime el tiempo (simula distintas velocidades de hélice)
    return librosa.effects.time_stretch(y=y.astype(np.float32),
                                        rate=float(rate)).astype(np.float32)


def mfcc_statistics(X):
    # De (N, bandas, T) a (N, bandas*5): media, std, min, max y mediana en el tiempo
    # Los clásicos (SVM/RF/KNN) necesitan un vector fijo por clip, así que se resume cada banda con cinco estadísticos temporales.
    estad = [X.mean(axis=2), X.std(axis=2), X.min(axis=2),
             X.max(axis=2), np.median(X, axis=2)]
    return np.concatenate(estad, axis=1).astype(np.float32)


try:
    import torch

    def spec_augment(x, n_mascaras_frec=2, ancho_frec=8, n_mascaras_tiempo=2, ancho_tiempo=6):
        # Pone a cero franjas aleatorias de frecuencia y tiempo del espectrograma (1, F, T).
        # Con los datos ya estandarizados, el cero es la media, así que enmascarar
        # equivale a "borrar" bandas, esto ayuda a que el modelo no se centre en coeficientes o
        # instantes concretos. Solo se utiliza en entrenamiento.
        _, F, T = x.shape
        for _ in range(n_mascaras_frec):
            w = int(torch.randint(0, ancho_frec + 1, (1,)).item())
            if 0 < w < F:
                f0 = int(torch.randint(0, F - w, (1,)).item())
                x[:, f0:f0 + w, :] = 0.0
        for _ in range(n_mascaras_tiempo):
            w = int(torch.randint(0, ancho_tiempo + 1, (1,)).item())
            if 0 < w < T:
                t0 = int(torch.randint(0, T - w, (1,)).item())
                x[:, :, t0:t0 + w] = 0.0
        return x

except ImportError:
    spec_augment = None


try:
    import torch
    from torch.utils.data import Dataset

    class RobustDroneDataset(Dataset):
        def __init__(self, waves, labels, cfg, noise_pool=None,
                     mean=None, std=None, train=False):
            self.waves = waves  # (N, L)
            self.labels = np.asarray(labels)
            self.cfg = cfg
            self.sr = cfg["audio"]["sample_rate"]
            self.seg = cfg["audio"]["segment_seconds"]
            self.noise_pool = noise_pool
            self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
            self.std = None if std is None else np.asarray(std, dtype=np.float32)
            self.train = train
            self.aug = cfg.get("augment", {}) or {}

        def __len__(self):
            return len(self.labels)

        def _augment_wave(self, y, label):
            a, rng = self.aug, np.random
            if (label == 1 and self.noise_pool is not None
                    and len(self.noise_pool) > 0
                    and rng.random() < a.get("noise_mix_prob", 0.0)):
                noise = self.noise_pool[rng.randint(len(self.noise_pool))]
                snr = rng.uniform(*a.get("snr_db_range", [0, 20]))
                y = mix_noise(y, noise, snr)
            if rng.random() < a.get("pitch_shift_prob", 0.0):
                try:
                    y = pitch_shift(y, self.sr, rng.uniform(*a.get("pitch_steps_range", [-2, 2])))
                except Exception:
                    pass
            if rng.random() < a.get("time_stretch_prob", 0.0):
                try:
                    y = time_stretch(y, rng.uniform(*a.get("time_stretch_range", [0.9, 1.1])))
                except Exception:
                    pass
            return y

        def __getitem__(self, i):
            y = np.asarray(self.waves[i], dtype=np.float32)
            label = int(self.labels[i])
            if self.train and self.aug.get("enable", False):
                y = self._augment_wave(y, label)
            y = fix_length(y, self.sr, self.seg)
            feat = waveform_to_features(y, self.cfg)    # (F, T)
            if self.mean is not None:
                feat = (feat - self.mean) / self.std
            x = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0).float()
            if self.train and self.aug.get("spec_augment", False):
                x = spec_augment(x)
            return x, label

except ImportError:
    RobustDroneDataset = None


AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


def _list_audio(dirs):
    files = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[aviso] carpeta no encontrada, se omite: {d}")
            continue
        for ext in AUDIO_EXT:
            files.extend(sorted(p.rglob(f"*{ext}")))
    return files


def _load_fixed(path, sr, seg):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return fix_length(y, sr, seg), len(y)


def preparar_datos(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    sr = cfg["audio"]["sample_rate"]
    seg = cfg["audio"]["segment_seconds"]

    drone_files = _list_audio(cfg["dataset"]["drone_dirs"])
    noise_files = _list_audio(cfg["dataset"]["noise_dirs"])
    # Topes opcionales para no reventar memoria (muestreo aleatorio)
    import random as _rnd
    _rnd.seed(cfg["seed"])
    max_d = cfg["dataset"].get("max_drone")
    max_n = cfg["dataset"].get("max_noise")
    if max_d and len(drone_files) > max_d:
        drone_files = _rnd.sample(drone_files, max_d)
    if max_n and len(noise_files) > max_n:
        noise_files = _rnd.sample(noise_files, max_n)
    print(f"Ficheros (tras topes): dron={len(drone_files)}  no-dron={len(noise_files)}")
    if not drone_files or not noise_files:
        raise SystemExit("Faltan ficheros de dron o de ruido. Revisar dataset.*_dirs.")

    files = drone_files + noise_files
    labels = np.array([1] * len(drone_files) + [0] * len(noise_files), dtype=np.int64)

    L = int(round(sr * seg))
    min_len = int(cfg["dataset"].get("min_seconds", 0.0) * sr)
    waves = np.zeros((len(files), L), dtype=np.float32)
    keep = np.ones(len(files), dtype=bool)
    n_short = 0
    for i, f in enumerate(tqdm(files, desc="Cargando y remuestreando")):
        try:
            fixed, rawlen = _load_fixed(f, sr, seg)
            if rawlen < min_len:
                keep[i] = False; n_short += 1; continue
            waves[i] = fixed
        except Exception as e:
            print(f"[aviso] no se pudo leer {f}: {e}")
            keep[i] = False
    if n_short:
        print(f"[info] descartados {n_short} clips por debajo de {cfg['dataset'].get('min_seconds', 0)}s")
    waves, labels = waves[keep], labels[keep]

    # Partición estratificada
    idx = np.arange(len(labels))
    strat = labels if cfg["split"]["stratify"] else None
    tr, te = train_test_split(idx, test_size=cfg["split"]["test_size"],
                              stratify=strat, random_state=cfg["seed"])
    strat_tr = labels[tr] if strat is not None else None
    rel_val = cfg["split"]["val_size"] / (1.0 - cfg["split"]["test_size"])
    tr, va = train_test_split(tr, test_size=rel_val,
                              stratify=strat_tr, random_state=cfg["seed"])

    def dist(ix):
        return {"total": int(len(ix)), "dron": int(labels[ix].sum()),
                "no_dron": int(len(ix) - labels[ix].sum())}
    print("Train:", dist(tr), "| Val:", dist(va), "| Test:", dist(te))

    # Conjunto de ruido para mezclas con clips de no-dron del TRAIN
    noise_idx = tr[labels[tr] == 0]
    cap = cfg["dataset"].get("max_noise_pool", 3000)
    if len(noise_idx) > cap:
        noise_idx = np.random.choice(noise_idx, cap, replace=False)
    noise_pool = waves[noise_idx].copy()
    print(f"Conjunto de ruido para mezclas: {noise_pool.shape[0]} clips")

    # Escalador log-mel: solo train, sin aumentos
    feats = [waveform_to_features(waves[i].astype(np.float32), cfg)
             for i in tqdm(tr, desc="Escalador (train)")]
    F = np.stack(feats) # (Ntr, n_mels, T)
    mean = F.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32) # (n_mels,1)
    std = (F.std(axis=(0, 2), keepdims=True)[0] + 1e-6).astype(np.float32)
    n_features, n_frames = F.shape[1], F.shape[2]

    # Pesos de clase
    n = len(tr); n_pos = int(labels[tr].sum()); n_neg = n - n_pos
    class_weight = {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}

    out = Path(cfg["dataset"]["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "waves.npy", waves.astype(np.float16))    # float16
    np.save(out / "labels.npy", labels)
    np.savez(out / "splits.npz", train=tr, val=va, test=te)
    np.save(out / "noise_pool.npy", noise_pool.astype(np.float16))
    np.savez(out / "scaler.npz", mean=mean, std=std)
    meta = {
        "sample_rate": sr, "segment_seconds": seg, "wave_len": L,
        "feature_type": cfg["features"]["type"],
        "n_features": int(n_features), "n_frames": int(n_frames),
        "features_cfg": cfg["features"], "audio": cfg["audio"],
        "counts": {"train": dist(tr), "val": dist(va), "test": dist(te)},
        "class_weight": class_weight, "seed": cfg["seed"],
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {out.resolve()}  |  feature map ({n_features}, {n_frames})")
    print(f"Pesos de clase: {class_weight}")


def _read_storage(val):
    # Lee (wave_mono_float32, sr) del almacenamiento crudo del audio
    if not isinstance(val, dict):
        raise ValueError("audio no es struct")
    if val.get("bytes"):
        y, sr = sf.read(io.BytesIO(val["bytes"]), dtype="float32")
    elif val.get("array") is not None:
        y = np.asarray(val["array"], dtype="float32")
        sr = int(val.get("sampling_rate") or 16000)
    elif val.get("path"):
        y, sr = sf.read(val["path"], dtype="float32")
    else:
        raise ValueError("struct de audio sin bytes/array/path")
    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(sr)


def _windows(y, sr, win_s, hop_s):
    if not win_s:
        yield y; return
    win = int(win_s * sr); hop = int((hop_s or win_s) * sr)
    if len(y) <= win:
        yield y; return
    for start in range(0, len(y) - win + 1, hop):
        yield y[start:start + win]


def _resolve_splits(ds_args, split):
    try:
        from datasets import get_dataset_split_names
        avail = get_dataset_split_names(*ds_args)
    except Exception:
        avail = []
    if split in avail:
        return [split]
    if avail:
        print(f"[info] split '{split}' no existe, se usan todos ({len(avail)}): {avail[:3]}...")
        return avail
    return [split]


def descargar_dataset(args):
    from datasets import load_dataset
    ds_args = [args.dataset] + ([args.name] if args.name else [])
    splits = _resolve_splits(ds_args, args.split)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for sp in splits:
        if args.max and n >= args.max:
            break
        ds = load_dataset(*ds_args, split=sp)
        label_names = None
        if not args.all_into:
            feat = ds.features.get(args.label_col)
            label_names = getattr(feat, "names", None)
        pa_table = ds.data.table    # almacenamiento crudo
        aidx = pa_table.schema.get_field_index(args.audio_col)
        lidx = pa_table.schema.get_field_index(args.label_col) if not args.all_into else -1
        for batch in pa_table.to_batches(max_chunksize=128):
            if args.max and n >= args.max:
                break
            audios = batch.column(aidx).to_pylist()
            labels = batch.column(lidx).to_pylist() if lidx >= 0 else [None] * len(audios)
            for a, lab in zip(audios, labels):
                if args.max and n >= args.max:
                    break
                try:
                    y, sr = _read_storage(a)
                except Exception as e:
                    print(f"[aviso] fila ilegible: {e}"); continue
                if args.all_into:
                    sub = out
                else:
                    name = label_names[lab] if (label_names and lab is not None) else str(lab)
                    sub = out / str(name); sub.mkdir(parents=True, exist_ok=True)
                for w in _windows(y, sr, args.window, args.hop):
                    if args.max and n >= args.max:
                        break
                    if len(w) < int(0.1 * sr):  # Descarta <0.1s
                        continue
                    sf.write(sub / f"{n:07d}.wav", w, sr)
                    n += 1
                    if n % 1000 == 0:
                        print(f"  {n} exportados...")
    print(f"Hecho: {n} ficheros en {out.resolve()}")
    if not args.all_into and label_names:
        print(f"Etiquetas: {label_names}  (mapea cada subcarpeta a drone_dirs o noise_dirs)")


def _cli():
    ap = argparse.ArgumentParser(description="Descarga y preparación de datos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preparar", help="preprocesa el dataset a uno listo para entrenar")
    p.add_argument("--config", default="config.yaml")

    d = sub.add_parser("descargar", help="pasa un dataset de Hugging Face a carpetas de wav")
    d.add_argument("--dataset", required=True)
    d.add_argument("--name", default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--split", default="train")
    d.add_argument("--audio-col", default="audio")
    d.add_argument("--label-col", default="label")
    d.add_argument("--all-into", default=None)
    d.add_argument("--window", type=float, default=None)
    d.add_argument("--hop", type=float, default=None)
    d.add_argument("--max", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "preparar":
        preparar_datos(args.config)
    else:
        descargar_dataset(args)


if __name__ == "__main__":
    _cli()
