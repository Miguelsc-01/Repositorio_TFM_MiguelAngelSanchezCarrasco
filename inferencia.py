# Inferencia en tiempo real, por micrófono o wav.

# En el PC se usa el backend ONNX y en la Raspberry Pi, la NPU Hailo. La captura
# del micrófono se adapta sola a la frecuencia que soporte el dispositivo y remuestrea
# a la del modelo, por lo que el mismo código vale en los dos dispositivos.

# Comandos:
    # python inferencia.py pc --model resnet --precision int8 --source mic --device 1
    # python inferencia.py pi --model resnet --source mic --device 1
    # python inferencia.py pc --model cnn_big --source dron.wav

from __future__ import annotations
import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from utiles import load_config
from datos import waveform_to_features, fix_length


def _softmax(x):
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class RollingBuffer:
    def __init__(self, n_samples):
        self.n = n_samples
        self.buf = np.zeros(n_samples, dtype=np.float32)
        self.filled = 0
        self._lock = threading.Lock()

    def push(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        with self._lock:
            k = len(x)
            if k >= self.n:
                self.buf = x[-self.n:].copy()
            else:
                self.buf = np.concatenate([self.buf[k:], x])
            self.filled = min(self.n, self.filled + k)

    def snapshot(self):
        with self._lock:
            return self.buf.copy(), self.filled >= self.n


def _resample(x, sr_in, sr_out):
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


class MicStream:
    # Captura del micrófono que se adapta a la frecuencia del dispositivo

    def __init__(self, sample_rate, window_samples, device=None, block_seconds=0.05):
        self.sr = sample_rate
        self.buffer = RollingBuffer(window_samples)
        self.device = device
        self._block_seconds = block_seconds
        self._stream = None

    def _resolver_frecuencia(self):
        import sounddevice as sd
        candidatas = []
        try:
            info = sd.query_devices(self.device, "input")
            candidatas.append(int(info["default_samplerate"]))
        except Exception:
            pass
        candidatas += [48000, 44100, 32000, 16000]
        for sr in candidatas:
            try:
                sd.check_input_settings(device=self.device, samplerate=sr, channels=1)
                return sr
            except Exception:
                continue
        return 48000

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self.buffer.push(_resample(mono, self._cap_sr, self.sr))

    def __enter__(self):
        import sounddevice as sd
        self._cap_sr = self._resolver_frecuencia()
        block = max(1, int(self._cap_sr * self._block_seconds))
        print(f"[mic] capturando a {self._cap_sr} Hz -> remuestreo a {self.sr} Hz")
        self._stream = sd.InputStream(
            samplerate=self._cap_sr, channels=1, blocksize=block,
            dtype="float32", callback=self._callback, device=self.device)
        self._stream.start()
        return self

    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def read_window(self):
        return self.buffer.snapshot()


class OnnxDetector:
    # Modelo ONNX en CPU (para el PC)

    def __init__(self, onnx_path, cfg, mean, std, threshold):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.cfg, self.mean, self.std, self.thr = cfg, mean, std, threshold
        self.sr = cfg["audio"]["sample_rate"]
        self.seg = cfg["audio"]["segment_seconds"]

    def predict(self, y):
        y = fix_length(y, self.sr, self.seg)
        feat = (waveform_to_features(y, self.cfg) - self.mean) / self.std
        x = feat[None, None].astype(np.float32)
        t0 = time.perf_counter()
        logits = self.sess.run(None, {self.input: x})[0]
        lat = (time.perf_counter() - t0) * 1000.0
        return float(_softmax(logits)[0, 1]), lat


class HailoDetector:
    # Modelo HEF en la NPU Hailo (para la Raspberry Pi)

    def __init__(self, hef_path, cfg, mean, std, threshold):
        from hailo_platform import (
            HEF, VDevice, HailoStreamInterface, InferVStreams,
            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
        )
        self.cfg = cfg
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1)
        self.thr = threshold
        self.sr = cfg["audio"]["sample_rate"]
        self.seg = cfg["audio"]["segment_seconds"]

        self._hef = HEF(hef_path)
        self._vdevice = VDevice()
        params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        self._ng = self._vdevice.configure(self._hef, params)[0]
        self._ng_params = self._ng.create_params()
        in_params = InputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32)
        self._in = self._hef.get_input_vstream_infos()[0]
        self._out = self._hef.get_output_vstream_infos()[0]
        self._in_shape = tuple(self._in.shape)
        print(f"[hailo] input '{self._in.name}' {self._in_shape} -> output '{self._out.name}'")
        self._pipe = InferVStreams(self._ng, in_params, out_params)
        self._pipe.__enter__()
        self._act = self._ng.activate(self._ng_params)
        self._act.__enter__()

    def predict(self, y):
        y = fix_length(y, self.sr, self.seg)
        feat = (waveform_to_features(y, self.cfg) - self.mean) / self.std
        x = np.ascontiguousarray(feat, dtype=np.float32).reshape((1,) + self._in_shape)
        t0 = time.perf_counter()
        res = self._pipe.infer({self._in.name: x})
        lat = (time.perf_counter() - t0) * 1000.0
        logits = np.asarray(res[self._out.name]).reshape(-1)
        return float(_softmax(logits)[0, 1]), lat

    def close(self):
        for cerrar in (lambda: self._act.__exit__(None, None, None),
                       lambda: self._pipe.__exit__(None, None, None),
                       lambda: self._vdevice.release()):
            try:
                cerrar()
            except Exception:
                pass


def _cargar_scaler(processed_dir):
    sc = np.load(Path(processed_dir) / "scaler.npz")
    mean = np.asarray(sc["mean"], dtype=np.float32).reshape(-1, 1)
    std = np.asarray(sc["std"], dtype=np.float32).reshape(-1, 1)
    return mean, std


def _cargar_umbral(model, default=0.5):
    for fname in (f"results_{model}.json", f"quantization_{model}.json"):
        p = Path("results") / fname
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                return float(json.load(fh).get("threshold", default))
    return default


def _dispositivo(dv):
    if dv is None:
        return None
    try:
        return int(dv)
    except (TypeError, ValueError):
        return dv


def _sobre_wav(det, wav_path, hop_seconds, smooth):
    import librosa
    y, _ = librosa.load(wav_path, sr=det.sr, mono=True)
    win = int(det.sr * det.seg)
    hop = max(1, int(det.sr * hop_seconds))
    hist = deque(maxlen=smooth)
    lats = []
    print(f"Simulando streaming sobre {wav_path}  ({len(y) / det.sr:.1f}s)\n")
    for start in range(0, max(1, len(y) - win + 1), hop):
        prob, lat = det.predict(y[start:start + win])
        lats.append(lat)
        hist.append(prob)
        avg = float(np.mean(hist))
        estado = "DRON" if avg >= det.thr else "no-dron"
        print(f"t={start / det.sr:5.2f}s  p(dron)={prob:.3f}  media={avg:.3f}  -> {estado:8s} ({lat:.1f} ms)")
    if lats:
        print(f"\nLatencia media/ventana: {np.mean(lats):.2f} ms")


def _por_micro(det, hop_seconds, smooth, device=None):
    win = int(det.sr * det.seg)
    hist = deque(maxlen=smooth)
    print("Escuchando por el micrófono... (Ctrl+C para parar)\n")
    with MicStream(det.sr, win, device=device) as mic:
        try:
            while True:
                time.sleep(hop_seconds)
                buf, ready = mic.read_window()
                if not ready:
                    continue
                prob, lat = det.predict(buf)
                hist.append(prob)
                avg = float(np.mean(hist))
                flag = "  DRON DETECTADO" if avg >= det.thr else "no-dron        "
                print(f"\rp(dron)={prob:.3f}  media={avg:.3f}  {flag}  ({lat:.1f} ms)   ",
                      end="", flush=True)
        except KeyboardInterrupt:
            print("\nParado.")


def _lanzar(args, backend):
    cfg = load_config(args.config)
    mean, std = _cargar_scaler(cfg["dataset"]["processed_dir"])
    thr = args.threshold if args.threshold is not None else _cargar_umbral(args.model)

    if backend == "pi":
        hef = Path("checkpoints") / f"{args.model}_hailo8.hef"
        if not hef.exists():
            raise FileNotFoundError(f"No existe {hef}.")
        det = HailoDetector(str(hef), cfg, mean, std, thr)
        print(f"Backend: HAILO NPU | {hef.name} | umbral={thr:.3f}\n")
    else:
        onnx = Path("checkpoints") / f"{args.model}_{args.precision}.onnx"
        if not onnx.exists():
            raise FileNotFoundError(f"No existe {onnx}. Ejecuta antes 'entrenar.py cuantizar'.")
        det = OnnxDetector(onnx, cfg, mean, std, thr)
        print(f"Backend: ONNX ({onnx.name}) | umbral={thr:.3f}\n")

    try:
        if args.source == "mic":
            _por_micro(det, args.hop, args.smooth, _dispositivo(args.device))
        else:
            _sobre_wav(det, args.source, args.hop, args.smooth)
    finally:
        cerrar = getattr(det, "close", None)
        if callable(cerrar):
            cerrar()


def _cli():
    ap = argparse.ArgumentParser(description="Inferencia en tiempo real (PC/Pi)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("pc", "pi"):
        s = sub.add_parser(nombre, help=f"inferencia en {'el PC (ONNX)' if nombre=='pc' else 'la Pi (NPU)'}")
        s.add_argument("--config", default="config.yaml")
        s.add_argument("--model", default="resnet",
                       choices=["cnn", "cnn_big", "resnet", "mobilenet"])
        s.add_argument("--source", default="mic", help="'mic' o ruta a un wav")
        s.add_argument("--hop", type=float, default=0.5)
        s.add_argument("--smooth", type=int, default=5)
        s.add_argument("--device", default=None)
        s.add_argument("--threshold", type=float, default=None)
        if nombre == "pc":
            s.add_argument("--precision", default="int8", choices=["fp32", "fp16", "int8"])
    args = ap.parse_args()
    _lanzar(args, args.cmd)


if __name__ == "__main__":
    _cli()
