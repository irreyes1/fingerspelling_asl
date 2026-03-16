# Studio Handoff

This branch is prepared so Google/Firebase Studio can inspect and improve the local web app for the ASL fingerspelling demo.

## What matters for Studio

- `src/web_api.py`
  - FastAPI app
  - route `/` for offline checkpoint testing
  - route `/webcam` for browser webcam + landmark demo + live backend inference
- `requirements.txt`
  - Python dependencies for the local backend
- `src/test_inference.py`
  - portable offline evaluation pipeline
- `src/offline_quick_eval.py`
  - cheap supplemental evaluation used by quick mode
- `src/realtime_webcam_infer.py`
  - reference desktop webcam pipeline used to align browser inference
- `src/model_loader.py`
  - reconstructs the model from checkpoint
- `src/data/dataset.py`
  - shared preprocessing for offline tests
- `src/models/`
  - model definitions

## What is intentionally omitted from the handoff

- datasets under `data/`
- virtual environments
- large generated artifacts
- heavy checkpoints and media unless already present in the main repo

## Current product shape

### Offline Test

- Runs a real Python evaluation from the web
- `quick` mode uses a cheap supplemental test path
- `extended` mode uses the heavier local validation path
- Shows:
  - CER
  - WER
  - Exact Match
  - Samples
  - GT/PRED examples

### Webcam

- Uses browser webcam
- Extracts hand landmarks in the browser with MediaPipe
- Sends landmark sequences to the backend
- Backend returns live predictions from the checkpoint
- Includes guided demo mode for the word `LAURA`

## Improvement goals for Studio

- Make the webcam page more polished and intuitive
- Improve the guided demo UX
- Improve layout and typography without breaking the current flow
- Keep the backend contract stable

## Backend contract

- `GET /health`
- `GET /`
- `POST /offline-test/run`
- `GET /webcam`
- `POST /webcam/infer`

## Run locally

```powershell
.venv\Scripts\python.exe -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/webcam
```
