Fingerspelling ASL

Proyecto para entrenamiento de reconocimiento de fingerspelling con CTC.

## 1) Clonar
```bash
git clone https://github.com/irreyes1/fingerspelling_asl.git
cd fingerspelling_asl
```

## 2) Crear entorno
### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Setup rapido (automatico):
```powershell
.\setup.ps1
```

### Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Datos (no incluidos en Git)
Este repo no sube parquets ni checkpoints pesados.

Estructura esperada:
```text
data/
  asl-fingerspelling/
    train.csv
    supplemental_metadata.csv
    character_to_prediction_index.json
    train_landmarks/*.parquet
    supplemental_landmarks/*.parquet
```

## 4) Entrenamiento
Comando base:
```bash
python -m src.train --data_dir data/asl-fingerspelling
```

Con W&B:
```bash
python -m src.train --data_dir data/asl-fingerspelling --use_wandb --wandb_project fingerspelling_asl
```

Flags utiles:
- `--wandb_entity <usuario_o_team>`
- `--wandb_run_name <nombre_run>`
- `--wandb_mode offline`
- `--wandb_tags tag1,tag2`

## 4.1) Arquitecturas en el repo
- `src/models/embedded_rnn.py`: baseline simple.
- `src/models/tcn_bilstm.py`: arquitectura del run final `archcmp2_tcn_bilstm_full_20260303`.
- `src/model_loader.py`: factory que detecta la arquitectura del checkpoint y carga el modelo correcto.

## 5) Evaluacion / Test

Evalua un checkpoint entrenado sobre el conjunto suplemental (participantes no vistos en entrenamiento — zero leakage).

> **Requisito:** el checkpoint debe haber sido entrenado **sin** `--use_supplemental` (si no, las metricas no son validas).

Comando base:
```bash
python -m src.evaluate \
    --ckpt artifacts/models/<checkpoint>.pt \
    --data_dir data/asl-fingerspelling
```

Flags utiles:
- `--batch_size <N>` (default: 64)
- `--num_workers <N>` (default: 0)
- `--n_examples <N>` — numero de ejemplos GT/PRED a imprimir (default: 10)

Metricas reportadas: **CER**, **WER**, **Exact Match**, **Avg Edit Distance**.

## 5.1) Inferencia rapida de checkpoint
```bash
python -m src.quick_infer --ckpt artifacts/models/<checkpoint>.pt --n 16
```

## 6) Webcam (MediaPipe)
```bash
python -m src.realtime_webcam
```

Nota: `src/realtime_webcam.py` usa `artifacts/models/hand_landmarker.task` para deteccion de manos.

Demo con checkpoint entrenado:
```bash
python -m src.realtime_webcam_infer --ckpt artifacts/models/archcmp2_tcn_bilstm_full_20260303_best.pt
```

Controles demo:
- `ESPACIO`: capturar letra puntual (modo guiado).
- `c`: limpiar `Live letters` y `Words`.
- `ESC`: salir.

## 7) Modelo entrenado (ejemplo)
Checkpoint best:
```text
artifacts/models/archcmp2_tcn_bilstm_full_20260303_best.pt
```

Copiar desde GCP a local:
```powershell
gcloud compute scp --zone=us-central1-b --project=buoyant-purpose-479417-t8 `
  instance-fingerspeling:~/fingerspelling_asl/artifacts/models/archcmp2_tcn_bilstm_full_20260303_best.pt `
  .\artifacts\models\
```

## 8) Troubleshooting
Guia rapida de errores frecuentes:

`docs/TROUBLESHOOTING.md`
