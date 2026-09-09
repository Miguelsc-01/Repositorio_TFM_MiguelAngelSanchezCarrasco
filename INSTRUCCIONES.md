# Detección acústica de drones con modelos cuantizados sobre NPU

Sistema para detectar drones por sonido en tiempo real. Se entrenan varios modelos
(CNNs, recurrentes y clásicos), se cuantizan a INT8 y las CNNs se despliegan en la
NPU Hailo-8 de una Raspberry Pi 5.

Tres dispositivos:
- PC (con GPU): preparar datos, entrenar, cuantizar y sacar gráficas.
- WSL2 (Linux dentro de Windows): compilar los modelos a HEF con el compilador de Hailo.
- Raspberry Pi 5 (con el AI HAT+): ejecutar en la NPU y medir latencia y consumo.

Todo se lanza desde la raíz del proyecto con `python <fichero>.py <comando>`.


# 1. Instalación (PC)

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Instalar PyTorch con el índice de CUDA que corresponda a la GPU.

GPU NVIDIA (Mi caso RTX 4070 Super):

    pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install datasets soundfile huggingface_hub    # solo para descargar datasets


# 2. Descargar los datasets

Cinco fuentes, unas de dron y otras de ruido. Las de Hugging Face vienen en parquet, así que `datos.py descargar` las pasa a carpetas de wav.

git clone https://github.com/saraalemadi/DroneAudioDataset.git data/raw/DroneAudioDataset

git clone https://github.com/karoldvl/ESC-50 data/raw/ESC-50

python datos.py descargar --dataset geronimobasso/drone-audio-detection-samples --out data/raw/geronimobasso --max 25000

python datos.py descargar --dataset ahlab-drone-project/DroneAudioSet --name drone-only --out data/raw/DroneAudioset/drone --all-into drone --window 1.0 --hop 1.0 --max 8000

python datos.py descargar --dataset ahlab-drone-project/DroneAudioSet --name drone-with-source --out data/raw/DroneAudioset/drone_source --all-into drone --window 1.0 --hop 1.0 --max 6000

En geronimobasso la etiqueta 1 = dron y 0 = no-dron. Las rutas de las carpetas ya están puestas en `config.yaml`, se deben ajustar si se cambian.


# 3. Preparar los datos

python datos.py preparar --config config.yaml


# 4. Entrenar

`--workers` = núcleos físicos de la CPU. Convergen pronto, con 30 épocas basta.

# CNNs (van a la NPU)
python entrenar.py red --model cnn       --workers 6 --epochs 30 --patience 6 --batch-size 128
python entrenar.py red --model cnn_big   --workers 6 --epochs 30 --patience 6 --batch-size 128
python entrenar.py red --model resnet    --workers 6 --epochs 30 --patience 6 --batch-size 128
python entrenar.py red --model mobilenet --workers 6 --epochs 30 --patience 6 --batch-size 128
# Recurrentes (solo CPU)
python entrenar.py red --model crnn --workers 6 --epochs 30 --patience 6 --batch-size 128
python entrenar.py red --model lstm --workers 6 --epochs 30 --patience 6 --batch-size 128
python entrenar.py red --model gru  --workers 6 --epochs 30 --patience 6 --batch-size 128
# Clásicos
python entrenar.py clasicos


# 5. Cuantizar (PC)

python entrenar.py cuantizar --config config.yaml


# 6. Gráficas y tablas (PC)

python analisis.py tabla    # precisión de todos los modelos
python analisis.py cuantizacion # tabla + figuras de cuantización
python analisis.py robustez-int8    # robustez frente a ruido, FP32 vs INT8
python analisis.py bench-pc # latencia de inferencia en el PC

Todo va a `results/` y `results/figuras/`.

# 6.1 Intervalos de confianza del PR-AUC (PC)

bootstrap sobre las predicciones ya guardadas en `results/scores_*.npz`.

python incertidumbre.py --figura


# 7. Compilar a HEF (WSL2)

El compilador de Hailo (Dataflow Compiler) solo corre en Linux x86, así que va en WSL2.

Abre WSL2 (`wsl -d Ubuntu-22.04`) y:

sudo apt update && sudo apt install

python -m venv ~/hailodfc          # el DFC pide Python 3.1X
source ~/hailodfc/bin/activate

# instala el Dataflow Compiler descargado de la Developer Zone de Hailo
pip install --upgrade pip
pip install ~/hailo_dataflow_compiler-*.whl
pip install pyyaml numpy    # lo que necesita compilar.py

# comprueba que responde
python -c "from hailo_sdk_client import ClientRunner; print('DFC OK')"

El `.whl` del DFC se baja del portal de desarrolladores de Hailo (no está en pip). Si el DFC no coge la GPU, se tiene que subir el umbral de memoria en su `nvidia_smi_gpu_selector.py` de 0.05 a 0.5.

En el PC genera antes la calibración:

python entrenar.py calibracion --config config.yaml

En WSL2, con el venv del DFC activado y desde la carpeta del proyecto (`/mnt/c/...`):

python compilar.py --model cnn
python compilar.py --model mobilenet
python compilar.py --model cnn_big
python compilar.py --model resnet

Cada uno saca `checkpoints/{modelo}_hailo8.hef` y su SNR de cuantización.


# 8. Desplegar en la Raspberry Pi 5

En la Pi (Raspberry Pi OS de 64 bits) hacen falta el HailoRT + PortAudio, y un venv con numpy < 2 (numpy 2.x rompe HailoRT).

sudo apt update && sudo apt install -y python3-venv libportaudio2 hailo-all
# 'hailo-all' instala el driver PCIe, el firmware y HailoRT del AI HAT+.
# Reinicia después:  sudo reboot

python3 -m venv ~/drone-venv
source ~/drone-venv/bin/activate

pip install --upgrade pip
pip install "numpy<2" librosa soundfile sounddevice onnxruntime pyyaml
# HailoRT para Python: instala el wheel que viene con hailo-all, p.ej.
pip install /usr/lib/python3/dist-packages/hailort-*.whl   # o el .whl de tu versión de HailoRT

# comprueba que la NPU responde
hailortcli fw-control identify        # debe listar el Hailo-8

En la Pi no se instala torch, la inferencia va por la NPU (HEF) y para la CPU, por ONNX. Por eso `datos.py` e `inferencia.py` funcionan sin PyTorch.

Copia a la Pi los HEF, el `scaler.npz` y `meta.json` de `data/processed/`, los `.py`, el `config.yaml`, los `results/results_*.json` (umbrales) y los `checkpoints/*_int8.onnx` (para el bench de CPU).

source ~/drone-venv/bin/activate

# benchmark de la NPU
hailortcli run checkpoints/resnet_hailo8.hef --batch-size 1 --measure-latency

# tiempo real por micrófono (ajusta --device a tu micro USB)
python inferencia.py pi --model resnet --source mic --device 1 --threshold 0.5 --smooth 5


# 9. Latencia y consumo en la Pi

python analisis.py bench-pi     # latencia NPU vs CPU (necesita los .onnx int8 copiados)
python analisis.py consumo      # consumo por software (PMIC de la Pi 5)

Copia `results/bench_pi.json` y `results/power_pi.json` al PC.


# 10. Comparativa final (PC)

python analisis.py final

Junta precisión + velocidad + consumo en `results/comparativa_final.md` y saca la
figura de Pareto en `results/figuras/`.

# Ficheros del proyecto

config.yaml     parámetros (audio, features, datasets, aumentos)
datos.py        descarga de datasets, preprocesado, features y aumentos
modelos.py      arquitecturas (CNNs, recurrentes, clásicos) y registro
entrenar.py     entrenamiento y cuantización INT8
compilar.py     compilación a HEF (se ejecuta en WSL con el DFC)
inferencia.py   tiempo real en el PC (ONNX) y en la Pi (NPU)
analisis.py     tablas, figuras, benchmarks y consumo
utiles.py       config, semilla y métricas

