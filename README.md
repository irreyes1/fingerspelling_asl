# Fingerspelling ASL Webcam Inference

Este repositorio ha quedado recortado para un objetivo concreto: ejecutar inferencia con webcam usando el mejor checkpoint del experimento remoto `39rmxai4` y validar localmente que el pipeline funciona.

## Estado actual del experimento

- Checkpoint principal: `artifacts/models/run_20260312_231052_best.pt`
- Run remoto asociado: `39rmxai4`
- Entrenamiento remoto observado: 50 epocas
- Mejor resultado remoto observado en el log:
  - `epoch 34`
  - `val CER = 0.3790`
  - `WER = 0.9109`
  - `ExactMatch = 0.0464`
  - `AvgEditDist = 4.9332`
- En `epoch 35` el run empeoro ligeramente (`val CER = 0.3799`), por eso el checkpoint bueno es el `best.pt`.

## Estructura minima que se conserva

- `src/realtime_webcam_infer.py`: inferencia en tiempo real con webcam.
- `src/model_loader.py`: reconstruccion automatica del modelo desde el checkpoint.
- `src/models/`: definicion de `EmbeddedRNN`, `TCNBiRNN` y subsampling temporal.
- `src/data/dataset.py`: lectura de parquets y preprocesado para el `TEST` offline.
- `src/test_inference.py`: fase `TEST` portable para validar assets, checkpoint y calidad offline.
- `tests/test_inference_smoke.py`: smoke test automatizado del pipeline de test.
- `artifacts/models/hand_landmarker.task`: detector de mano de MediaPipe.
- `artifacts/models/run_20260312_231052_best.pt`: checkpoint a usar.
- `artifacts/eval/latest_portable_test/`: resumen, casos GT/PRED y graficas del test local.

## Como funciona la inferencia con webcam

El flujo de `src/realtime_webcam_infer.py` es este:

1. Carga el checkpoint con `load_model_from_checkpoint`.
2. Reconstruye la arquitectura inferida desde el `state_dict`.
3. Lee la configuracion embebida en el checkpoint para deducir:
   - `input_dim`
   - `max_frames`
   - si necesita deltas (`input_dim > 63` en este checkpoint, por tanto usa 126 features)
   - si debe aplicar filtros de texto
4. Abre MediaPipe Hand Landmarker y la webcam.
5. En cada frame:
   - detecta manos
   - se queda con la mano derecha; si no hay etiqueta fiable, usa la primera mano
   - convierte 21 landmarks a un vector de 63 valores `(x, y, z)`
   - centra la mano en la muneca
   - escala por `median_radius` para reducir variaciones de distancia a camara
6. Si el checkpoint espera 126 features, concatena `delta = frame_actual - frame_anterior`.
7. Mantiene un buffer temporal de hasta `max_frames`.
8. Cada `infer_every` frames ejecuta el modelo sobre el buffer.
9. Decodifica con CTC greedy y aplica un filtro temporal adicional para tiempo real:
   - `vote_window`
   - `letter_conf_threshold`
   - `min_vote_conf`
   - `min_margin`
   - `blank_skip_threshold`
10. Usa una maquina de estados simple:
   - `stable_required` para aceptar una letra
   - `release_frames` para volver a armar el sistema
   - `pause_frames` para consolidar una palabra
11. Muestra en overlay:
   - top-3 clases
   - confianza
   - blank probability
   - margin
   - texto CTC bruto
   - letras vivas y palabras comprometidas

## Filtros y preprocesado aplicados

En este ultimo checkpoint, el `TEST` local detecta y usa:

- `expected_input_dim = 126`
- `use_delta_features = true`
- `normalize_landmarks = true`
- `landmark_scale_mode = median_radius`
- `lowercase_phrases = false`

Eso significa:

1. Se toman solo landmarks de la mano derecha.
2. Se centra cada frame respecto a la muneca.
3. Se normaliza la escala de la mano por radio mediano 2D.
4. Se anaden deltas temporales, pasando de 63 a 126 dimensiones.
5. No se fuerzan minusculas en la evaluacion local.

## Fase TEST

La fase `TEST` comprueba tres cosas:

1. Que existen los assets necesarios:
   - checkpoint
   - `hand_landmarker.task`
   - vocabulario JSON
   - parquets locales
2. Que el checkpoint y el pipeline de datos son compatibles:
   - mismas dimensiones de entrada
   - mismo preprocesado
3. Que el modelo produce predicciones sobre una particion local determinista y deja:
   - `summary.json`
   - `cases.csv`
   - graficas PNG

### Comando de test offline

```bash
python -m src.test_inference --output_dir artifacts/eval/latest_portable_test
```

### Smoke test automatizado

```bash
python -m unittest tests.test_inference_smoke -v
```

### Comprobacion opcional de webcam

```bash
python -m src.test_inference --run_webcam_check --camera_index 0
```

## Como lanzar la inferencia con webcam

```bash
python -m src.realtime_webcam_infer ^
  --ckpt artifacts/models/run_20260312_231052_best.pt ^
  --hand_model artifacts/models/hand_landmarker.task ^
  --vocab_json data/asl-fingerspelling/character_to_prediction_index.json
```

Controles en la ventana:

- `ESC`: salir
- `SPACE`: capturar letra en modo guiado
- `c`: limpiar texto acumulado

## Datos usados en este portatil

El portatil no contiene el dataset completo. La evaluacion local se ha hecho solo con los parquets presentes en `data/asl-fingerspelling/train_landmarks`.

Cobertura local medida por `src.test_inference.py`:

- Filas totales en `train.csv`: `67,208`
- Parquets locales disponibles: `4`
- Filas cuyo `file_id` existe en local: `287`
- Cobertura local sobre el CSV total: `0.427%`
- Participantes cubiertos en local: `84`

Split de test local:

- Train local: `237` filas
- Validation local: `50` filas
- Casos validos realmente evaluables: `32`
- Casos invalidados por `input_shorter_than_target`: `18`

## Evaluacion escrita del ultimo checkpoint

### 1. Resultado del entrenamiento remoto

El resultado relevante del experimento bueno no es el test del portatil, sino el mejor punto observado en remoto:

- `val CER = 0.3790`
- `WER = 0.9109`
- `ExactMatch = 0.0464`
- `AvgEditDist = 4.9332`

Eso indica que el checkpoint remoto si aprendio una senal util sobre el split grande usado alli.

### 2. Resultado del TEST local portable

Metrica sobre el subconjunto disponible en este portatil:

- `CER = 0.9488`
- `WER = 1.0678`
- `ExactMatch = 0.0000`
- `AvgEditDistance = 15.0000`

Interpretacion:

- El pipeline funciona tecnicamente: carga assets, reconstruye el modelo, evalua y produce predicciones.
- El resultado cuantitativo local es malo.
- No debe interpretarse como metrica final del modelo.

Motivos:

1. El portatil solo tiene `4` parquets del dataset.
2. Eso representa solo `0.427%` de las filas de `train.csv`.
3. El subconjunto local esta sesgado hacia nombres, telefonos, direcciones y cadenas largas mixtas.
4. El checkpoint remoto se selecciono usando una validacion mucho mayor y mas estable que la disponible en local.

## Evaluacion grafica

Se generan estas graficas:

- `artifacts/eval/latest_portable_test/metrics_overview.png`
- `artifacts/eval/latest_portable_test/cer_histogram.png`
- `artifacts/eval/latest_portable_test/edit_distance_vs_gt_len.png`

Resumen de lo que muestran:

- `metrics_overview.png`: vista agregada de CER medio, distancia de edicion media, exact match y longitud media.
- `cer_histogram.png`: distribucion del error por caso.
- `edit_distance_vs_gt_len.png`: como crece el error absoluto con la longitud del GT.

## Ejemplos GT/PRED

Los casos completos estan en `artifacts/eval/latest_portable_test/cases.csv`.

### Mejores casos locales

| GT | PRED | CER |
| --- | --- | --- |
| `+962-11-5003-6049-58` | `3)5%7 -:3:40/-` | `0.8000` |
| `172-253-5090` | `-3 ` | `0.8333` |
| `+58-32-2543-52-117` | `5!3#:33` | `0.8333` |
| `+52-43-1780-2946` | `--:33 #:5` | `0.8750` |
| `3416 4475` | `4!)3:` | `0.8889` |

### Peores casos locales

| GT | PRED | CER |
| --- | --- | --- |
| `dvayurcare.in/temperatura` | `34)1` | `1.0000` |
| `budgetblindswichitaks` | `3` | `1.0000` |
| `453 east 94th avenue` | `#/5%//!` | `1.0000` |
| `nalvamartinscampos` | `/4)!5! 6355):4` | `1.0000` |
| `naturalcouture.jp` | `34)5-3/5` | `1.0000` |

## Limitaciones actuales

- La validacion portable no sustituye a la validacion remota completa.
- El smoke test no abre la webcam; solo verifica pipeline y artefactos.
- La comprobacion de webcam real debe hacerse en la maquina del usuario con `--run_webcam_check`.
- No se han descargado `test_landmarks` de Kaggle porque no hacen falta para verificar el sistema y la cobertura local ya permite test funcional.

## Resultado practico

El repo queda preparado para:

1. Cargar el checkpoint bueno.
2. Ejecutar inferencia con webcam.
3. Pasar un `TEST` portable y reproducible.
4. Consultar documentacion y resultados del ultimo experimento en un solo sitio.
