# Compilación a HEF para Hailo-8 (se ejecuta en WSL2 con el DFC)

# Va aparte del resto porque en WSL solo tenemos el compilador de Hailo: este fichero no importa torch, librosa ni sklearn, solo numpy y el SDK de Hailo.
# La calibración se lee del .npy que se genera antes en el PC con 'entrenar.py calibracion'.

# Uso (en WSL, venv del DFC activado, desde la carpeta del proyecto):
    # python compilar.py --model cnn
    # python compilar.py --model resnet

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main(args):
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    processed = Path(cfg["dataset"]["processed_dir"])
    with open(processed / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    n_bandas, n_frames = meta["n_features"], meta["n_frames"]

    calib_path = Path("checkpoints") / "calib_nhwc.npy"
    if not calib_path.exists():
        raise FileNotFoundError(
            f"No existe {calib_path}. Generar en el PC con "
            f"'python entrenar.py calibracion' y cópialo a WSL.")
    calib = np.load(calib_path)[: args.calib].astype(np.float32)
    print(f"Calibración: {calib.shape}  (rango [{calib.min():.2f}, {calib.max():.2f}])")

    onnx_path = f"checkpoints/{args.model}_fp32.onnx"
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"No existe {onnx_path}. Copiar desde el PC.")

    from hailo_sdk_client import ClientRunner
    runner = ClientRunner(hw_arch=args.hw_arch)

    print(f"Parseando {onnx_path} ...")
    runner.translate_onnx_model(
        onnx_path, args.model,
        start_node_names=[args.input_name], end_node_names=[args.output_name],
        net_input_shapes={args.input_name: [1, 1, n_bandas, n_frames]},
    )
    runner.save_har(f"checkpoints/{args.model}_parsed.har")

    print("Optimizando + cuantizando (INT8) con la calibración ...")
    runner.optimize(calib)
    runner.save_har(f"checkpoints/{args.model}_quantized.har")

    print("Compilando a HEF ...")
    hef = runner.compile()
    salida = f"checkpoints/{args.model}_hailo8.hef"
    with open(salida, "wb") as fh:
        fh.write(hef)
    print(f"\nHEF generado: {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="resnet",
                    choices=["cnn", "cnn_big", "resnet", "mobilenet"])
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--input-name", default="input")
    ap.add_argument("--output-name", default="logits")
    ap.add_argument("--calib", type=int, default=1024)
    main(ap.parse_args())
