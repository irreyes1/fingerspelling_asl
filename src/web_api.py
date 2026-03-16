import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List

from src.model_loader import load_model_from_checkpoint
from src.realtime_webcam_infer import (
    build_compact_letter_vocab,
    build_model_input_from_history,
    ctc_decode_text,
    sanitize_decoded_text,
)
from src.test_inference import run_test


class OfflineTestRequest(BaseModel):
    ckpt: str = "artifacts/models/run_20260312_231052_best.pt"
    data_dir: str = "data/asl-fingerspelling"
    hand_model: str = "artifacts/models/hand_landmarker.task"
    mode: str = "quick"
    max_samples: int = Field(default=100, ge=1)
    batch_size: int = Field(default=16, ge=1)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    split_csv: Optional[str] = None


class WebcamInferRequest(BaseModel):
    frames: List[List[float]]
    handedness: Optional[str] = None
    ckpt: str = "artifacts/models/run_20260312_231052_best.pt"
    min_frames: int = Field(default=12, ge=1)


_WEBCAM_MODEL_CACHE: dict = {}


def _resolve_request_settings(request: OfflineTestRequest) -> tuple[int, int]:
    return int(request.max_samples), max(int(request.batch_size), 16)


def _summary_to_web_json(summary: dict, mode: str, max_samples: int) -> dict:
    metrics = summary.get("metrics", {})
    split = summary.get("split", {})
    examples = summary.get("best_examples", [])[:4]
    return {
        "checkpoint": Path(summary.get("checkpoint", "")).name,
        "mode": str(mode),
        "maxSamples": int(max_samples),
        "metrics": {
            "cer": float(metrics.get("cer", 0.0)),
            "wer": float(metrics.get("wer", 0.0)),
            "exactMatch": float(metrics.get("sequence_accuracy", 0.0)),
            "samples": int(split.get("val_cases_scored", 0)),
        },
        "examples": [
            {
                "gt": str(item.get("gt", "")),
                "pred": str(item.get("pred", "")),
                "cer": float(item.get("cer", 0.0)),
            }
            for item in examples
        ],
    }


def _get_cached_webcam_runtime(ckpt_path: str) -> dict:
    key = str(Path(ckpt_path).resolve())
    cached = _WEBCAM_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_model_from_checkpoint(key, device=device)
    ckpt_cfg = loaded.config or {}
    expected_dim = int(loaded.input_dim)
    expected_output_dim = int(loaded.output_dim)
    _, idx2char, blank_id = build_compact_letter_vocab()
    runtime = {
        "device": device,
        "model": loaded.model,
        "config": ckpt_cfg,
        "expected_dim": expected_dim,
        "expected_output_dim": expected_output_dim,
        "use_delta_features": bool(ckpt_cfg.get("use_delta_features", expected_dim > 63)),
        "normalize_landmarks": not bool(ckpt_cfg.get("disable_landmark_centering", False)),
        "lowercase": bool(ckpt_cfg.get("lowercase_phrases", expected_output_dim == 28)),
        "letters_only": bool(ckpt_cfg.get("letters_only", expected_output_dim == 28)),
        "max_frames": int(ckpt_cfg.get("max_frames", 160)),
        "idx2char": idx2char,
        "blank_id": blank_id,
    }
    _WEBCAM_MODEL_CACHE[key] = runtime
    return runtime


def _run_pr3_quick_eval(request: OfflineTestRequest) -> dict:
    worktree = Path(r"C:\ENTORNO\fingerspelling_asl_pr3_review")
    python_exe = Path(r"C:\ENTORNO\fingerspelling_asl\.venv\Scripts\python.exe")
    command = [
        str(python_exe),
        "-m",
        "src.evaluate",
        "--ckpt",
        str(Path(request.ckpt).resolve()),
        "--data_dir",
        str(Path(request.data_dir).resolve()),
        "--batch_size",
        str(max(int(request.batch_size), 16)),
        "--n_examples",
        "4",
        "--num_workers",
        "0",
        "--max_files",
        "3",
        "--max_samples",
        str(int(request.max_samples)),
    ]
    completed = subprocess.run(
        command,
        cwd=str(worktree),
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout

    samples_match = re.search(r"Samples\s*:\s*(\d+)", stdout)
    cer_match = re.search(r"CER\s*:\s*([0-9.]+)", stdout)
    wer_match = re.search(r"WER\s*:\s*([0-9.]+)", stdout)
    exact_match = re.search(r"Exact Match\s*:\s*([0-9.]+)", stdout)
    example_matches = re.findall(
        r"\[..\]\s+CER=([0-9.]+)\s+GT:\s+'([^']*)'\s+PRED:\s+'([^']*)'",
        stdout,
    )

    return {
        "status": "ok",
        "message": f'Offline evaluation completed with mode="quick" and maxSamples={int(request.max_samples)}',
        "result": {
            "checkpoint": Path(request.ckpt).name,
            "mode": "quick",
            "maxSamples": int(request.max_samples),
            "metrics": {
                "cer": float(cer_match.group(1)) if cer_match else 0.0,
                "wer": float(wer_match.group(1)) if wer_match else 0.0,
                "exactMatch": float(exact_match.group(1)) if exact_match else 0.0,
                "samples": int(samples_match.group(1)) if samples_match else 0,
            },
            "examples": [
                {
                    "cer": float(cer),
                    "gt": gt,
                    "pred": pred,
                }
                for cer, gt, pred in example_matches[:4]
            ],
        },
        "artifacts": {
            "stdout_source": str(worktree / "src" / "evaluate.py"),
        },
    }


app = FastAPI(title="Fingerspelling Web API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def offline_test_app() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fingerspelling Offline Test</title>
  <style>
    :root {
      --bg: #f5f5f4;
      --panel: #ffffff;
      --line: #e7e5e4;
      --text: #1c1917;
      --muted: #78716c;
      --accent: #b45309;
      --accent-soft: #fef3c7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 24px 56px;
    }
    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
    }
    h1 {
      font-size: 48px;
      line-height: 1.05;
      margin: 14px 0 0;
    }
    .lead {
      max-width: 760px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.7;
      margin-top: 18px;
    }
    .banner {
      display: none;
      margin-top: 22px;
      padding: 14px 16px;
      border: 1px solid #fcd34d;
      background: var(--accent-soft);
      border-radius: 16px;
      font-size: 14px;
    }
    .grid {
      display: grid;
      gap: 24px;
      grid-template-columns: 340px 1fr;
      margin-top: 32px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
    }
    .panel h2 {
      margin: 0;
      font-size: 20px;
    }
    label {
      display: block;
      margin: 16px 0 8px;
      font-size: 14px;
      font-weight: 600;
      color: #44403c;
    }
    input, select, button {
      width: 100%;
      border-radius: 16px;
      border: 1px solid #d6d3d1;
      padding: 12px 14px;
      font-size: 14px;
    }
    button {
      margin-top: 18px;
      border: 0;
      background: #1c1917;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .cards {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
    }
    .card .k {
      color: var(--muted);
      font-size: 13px;
    }
    .card .v {
      margin-top: 12px;
      font-size: 34px;
      font-weight: 700;
    }
    .bars {
      display: grid;
      gap: 16px;
      margin-top: 20px;
    }
    .bar-row-head {
      display: flex;
      justify-content: space-between;
      font-size: 14px;
      margin-bottom: 6px;
    }
    .bar-bg {
      height: 12px;
      background: #e7e5e4;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 12px;
      background: var(--accent);
      border-radius: 999px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
      font-size: 18px;
    }
    th, td {
      padding: 16px 18px;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    thead th {
      background: #f5f5f4;
      color: var(--muted);
      border-top: 0;
      font-size: 16px;
      font-weight: 700;
    }
    .cer-cell {
      width: 120px;
      font-weight: 700;
      color: var(--accent);
    }
    .text-gt, .text-pred {
      font-size: 24px;
      line-height: 1.35;
    }
    .stack {
      display: grid;
      gap: 24px;
    }
    .small {
      color: var(--muted);
      font-size: 13px;
    }
    .nav {
      display: flex;
      gap: 12px;
      margin-top: 18px;
    }
    .nav a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 150px;
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid #d6d3d1;
      color: var(--text);
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
      background: white;
    }
    .nav a.primary {
      background: #1c1917;
      color: white;
      border-color: #1c1917;
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      h1 { font-size: 40px; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <div class="eyebrow">ASL Fingerspelling</div>
    <h1>Offline checkpoint test</h1>
    <p class="lead">
      Launch a real Python evaluation from the browser, inspect CER/WER metrics, and review GT/PRED examples from the current checkpoint.
    </p>
    <div class="nav">
      <a href="/" class="primary">Offline Test</a>
      <a href="/webcam">Webcam</a>
    </div>
    <div id="banner" class="banner"></div>

    <section class="grid">
      <div class="panel">
        <h2>Test configuration</h2>
        <label for="mode">Test mode</label>
        <select id="mode">
          <option value="quick">Quick test</option>
          <option value="extended">Extended test</option>
        </select>

        <label for="maxSamples">Max samples</label>
        <input id="maxSamples" type="number" min="1" value="100" />

        <button id="runButton" type="button">Run evaluation</button>
        <p class="small" style="margin-top: 12px;">
          The API runs the real Python test and stores artifacts in <code>artifacts/eval/web_runs</code>.
        </p>
      </div>

      <div class="stack">
        <div class="cards">
          <div class="card"><div class="k">CER</div><div id="cer" class="v">-</div></div>
          <div class="card"><div class="k">WER</div><div id="wer" class="v">-</div></div>
          <div class="card"><div class="k">Exact Match</div><div id="exactMatch" class="v">-</div></div>
          <div class="card"><div class="k">Samples</div><div id="samples" class="v">-</div></div>
        </div>

        <div class="panel">
          <h2>Metrics overview</h2>
          <div class="bars">
            <div>
              <div class="bar-row-head"><span>CER</span><span id="cerLabel">-</span></div>
              <div class="bar-bg"><div id="cerBar" class="bar-fill" style="width: 0%;"></div></div>
            </div>
            <div>
              <div class="bar-row-head"><span>WER</span><span id="werLabel">-</span></div>
              <div class="bar-bg"><div id="werBar" class="bar-fill" style="width: 0%;"></div></div>
            </div>
            <div>
              <div class="bar-row-head"><span>Exact Match</span><span id="exactLabel">-</span></div>
              <div class="bar-bg"><div id="exactBar" class="bar-fill" style="width: 0%;"></div></div>
            </div>
          </div>
        </div>

        <div class="panel">
          <h2>GT / PRED examples</h2>
          <table>
            <thead>
              <tr><th style="width: 120px;">CER</th><th>GT</th><th>PRED</th></tr>
            </thead>
            <tbody id="examplesBody">
              <tr><td colspan="3" class="small">No results yet.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <script>
    const banner = document.getElementById("banner");
    const runButton = document.getElementById("runButton");
    const modeInput = document.getElementById("mode");
    const maxSamplesInput = document.getElementById("maxSamples");

    function setBanner(message, visible = true) {
      banner.textContent = message;
      banner.style.display = visible ? "block" : "none";
    }

    function setMetric(id, value) {
      document.getElementById(id).textContent = value;
    }

    function setBar(barId, labelId, value, max) {
      const safe = Number.isFinite(value) ? value : 0;
      const pct = Math.max(0, Math.min((safe / max) * 100, 100));
      document.getElementById(barId).style.width = pct + "%";
      document.getElementById(labelId).textContent = safe.toFixed(4);
    }

    function renderExamples(examples) {
      const body = document.getElementById("examplesBody");
      if (!examples || examples.length === 0) {
        body.innerHTML = '<tr><td colspan="3" class="small">No examples returned.</td></tr>';
        return;
      }
      body.innerHTML = examples.map((item) => `
        <tr>
          <td class="cer-cell">${Number(item.cer).toFixed(2)}</td>
          <td class="text-gt">${item.gt}</td>
          <td class="text-pred">${item.pred}</td>
        </tr>
      `).join("");
    }

    function renderResult(result) {
      setMetric("cer", result.metrics.cer.toFixed(4));
      setMetric("wer", result.metrics.wer.toFixed(4));
      setMetric("exactMatch", result.metrics.exactMatch.toFixed(4));
      setMetric("samples", String(result.metrics.samples));
      setBar("cerBar", "cerLabel", result.metrics.cer, 1);
      setBar("werBar", "werLabel", result.metrics.wer, 1.2);
      setBar("exactBar", "exactLabel", result.metrics.exactMatch, 1);
      renderExamples(result.examples);
    }

    async function runEvaluation() {
      runButton.disabled = true;
      runButton.textContent = "Running...";
      setBanner("Launching real Python evaluation...");

      try {
        const response = await fetch("/offline-test/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode: modeInput.value,
            max_samples: Number(maxSamplesInput.value || 100)
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.error || "Request failed");
        }

        renderResult(data.result);
        setBanner(data.message);
      } catch (error) {
        setBanner("Error: " + error.message);
      } finally {
        runButton.disabled = false;
        runButton.textContent = "Run evaluation";
      }
    }

    runButton.addEventListener("click", runEvaluation);
  </script>
</body>
</html>
    """


@app.get("/webcam", response_class=HTMLResponse)
def webcam_app() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fingerspelling Webcam</title>
  <style>
    :root {
      --bg: #f5f5f4;
      --panel: #ffffff;
      --line: #e7e5e4;
      --text: #1c1917;
      --muted: #78716c;
      --accent: #b45309;
      --accent-soft: #fef3c7;
      --good: #166534;
      --bad: #991b1b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1280px;
      margin: 0 auto;
      padding: 40px 24px 56px;
    }
    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
    }
    h1 {
      font-size: 48px;
      line-height: 1.05;
      margin: 14px 0 0;
    }
    .lead {
      max-width: 760px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.7;
      margin-top: 18px;
    }
    .nav {
      display: flex;
      gap: 12px;
      margin-top: 18px;
    }
    .nav a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 150px;
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid #d6d3d1;
      color: var(--text);
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
      background: white;
    }
    .nav a.primary {
      background: #1c1917;
      color: white;
      border-color: #1c1917;
    }
    .banner {
      display: none;
      margin-top: 22px;
      padding: 14px 16px;
      border: 1px solid #fcd34d;
      background: var(--accent-soft);
      border-radius: 16px;
      font-size: 14px;
    }
    .grid {
      display: grid;
      gap: 24px;
      grid-template-columns: 1.2fr 0.8fr;
      margin-top: 32px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
    }
    .panel h2 {
      margin: 0;
      font-size: 20px;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      margin-top: 18px;
      flex-wrap: wrap;
    }
    button {
      border-radius: 999px;
      border: 0;
      padding: 12px 18px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      background: #1c1917;
      color: white;
    }
    button.secondary {
      background: white;
      color: var(--text);
      border: 1px solid #d6d3d1;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .video-shell {
      margin-top: 18px;
      border-radius: 24px;
      overflow: hidden;
      background: #d6d3d1;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .camera-target {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      background: #fafaf9;
    }
    .camera-target img {
      width: 100%;
      max-width: 320px;
      display: block;
      margin: 12px auto 0;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
    }
      video, canvas {
        width: 100%;
        height: 100%;
        object-fit: cover;
        background: black;
      }
      #overlay {
        position: absolute;
        inset: 0;
        pointer-events: none;
      }
      #snapshotCanvas { display: none; }
      .placeholder {
        color: #57534e;
        font-size: 18px;
    }
    .cards {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
    }
    .k {
      color: var(--muted);
      font-size: 13px;
    }
    .v {
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
    }
    .mono {
      font-family: Consolas, monospace;
      font-size: 20px;
      line-height: 1.5;
      word-break: break-word;
    }
    .pill.good { color: var(--good); }
    .pill.bad { color: var(--bad); }
    .hint {
      margin-top: 14px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .demo-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr;
      margin-top: 16px;
    }
    .demo-word {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .demo-letter {
      min-width: 58px;
      height: 58px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #fafaf9;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
      font-weight: 700;
    }
    .demo-letter.active {
      background: #1c1917;
      color: white;
      border-color: #1c1917;
    }
    .demo-letter.done {
      background: #166534;
      color: white;
      border-color: #166534;
    }
    .target-card {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      background: #fafaf9;
    }
    .target-card img {
      width: 100%;
      max-width: 360px;
      display: block;
      margin: 14px auto 0;
      border-radius: 18px;
      background: white;
      border: 1px solid var(--line);
    }
    .target-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      flex-wrap: wrap;
    }
    .target-letter {
      font-size: 44px;
      font-weight: 700;
      color: var(--accent);
    }
    .target-step {
      color: var(--muted);
      font-size: 14px;
    }
    .source-link {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .source-link a {
      color: var(--accent);
    }
    img.snapshot {
      margin-top: 18px;
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      display: none;
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 40px; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <div class="eyebrow">ASL Fingerspelling</div>
    <h1>Webcam test</h1>
    <p class="lead">
      Open the browser camera, verify capture quality, and prepare the live path that we will later connect to model inference.
    </p>
    <div class="nav">
      <a href="/">Offline Test</a>
      <a href="/webcam" class="primary">Webcam</a>
    </div>
    <div id="banner" class="banner"></div>

    <section class="grid">
      <div class="panel">
        <h2>Camera preview</h2>
          <div class="toolbar">
            <button id="startButton" type="button">Start webcam</button>
            <button id="stopButton" type="button" class="secondary" disabled>Stop webcam</button>
            <button id="captureButton" type="button" class="secondary" disabled>Capture frame</button>
            <button id="captureLetterButton" type="button" class="secondary" disabled>Capture prediction</button>
            <button id="resetDemoButton" type="button" class="secondary">Reset demo</button>
          </div>
          <div class="camera-target">
            <div class="target-head">
              <div>
                <div class="k">Current target next to camera</div>
                <div id="cameraTargetLetter" class="target-letter">L</div>
              </div>
              <div id="cameraTargetStep" class="target-step">Step 1 / 5</div>
            </div>
            <img id="cameraTargetImage" alt="ASL target handshape near camera" />
            <div id="cameraTargetHint" class="hint"></div>
          </div>
          <div class="video-shell">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="overlay"></canvas>
            <canvas id="snapshotCanvas"></canvas>
            <div id="placeholder" class="placeholder">Webcam preview will appear here.</div>
          </div>
          <img id="snapshot" class="snapshot" alt="Last captured frame" />
        <p class="hint">
          This first version validates webcam access in the browser. The next step will be to connect the live stream to landmark extraction and model inference.
        </p>
      </div>

        <div>
        <div class="cards">
          <div class="card">
            <div class="k">Camera access</div>
            <div id="cameraAccess" class="v pill bad">Not started</div>
          </div>
          <div class="card">
            <div class="k">Stream state</div>
            <div id="streamState" class="v">Idle</div>
          </div>
            <div class="card">
              <div class="k">Captured frames</div>
              <div id="capturedFrames" class="v">0</div>
            </div>
            <div class="card">
              <div class="k">Hand detected</div>
              <div id="handDetected" class="v pill bad">No</div>
            </div>
            <div class="card">
              <div class="k">Detected side (mirrored camera)</div>
              <div id="handedness" class="v">-</div>
            </div>
            <div class="card">
              <div class="k">Landmarks</div>
              <div id="landmarkCount" class="v">0</div>
            </div>
            <div class="card">
              <div class="k">Top prediction</div>
              <div id="topPrediction" class="v">-</div>
            </div>
            <div class="card">
              <div class="k">Last action</div>
              <div id="lastAction" class="v" style="font-size:22px;">Waiting</div>
            </div>
          </div>

          <div class="panel" style="margin-top:24px;">
            <h2>Decoded text</h2>
            <p class="hint">Live predictions now come from the Python backend using browser landmarks.</p>
            <div class="card" style="margin-top:18px;">
              <div class="k">Raw prediction</div>
              <div id="rawPrediction" class="v mono">-</div>
            </div>
            <div class="card" style="margin-top:16px;">
              <div class="k">Top-3</div>
              <div id="top3" class="v mono">-</div>
            </div>
            <div class="card" style="margin-top:16px;">
              <div class="k">Captured letters</div>
              <div id="capturedLetters" class="v mono">-</div>
          </div>
            <div class="card" style="margin-top:16px;">
              <div class="k">Words</div>
              <div id="words" class="v mono">-</div>
            </div>
          </div>

          <div class="panel" style="margin-top:24px;">
            <h2>Demo word: LAURA</h2>
            <p class="hint">
              Follow the sequence below. The demo only advances to the next letter when the current target is detected repeatedly and stably.
            </p>
            <div id="demoWord" class="demo-word"></div>
            <div class="target-card">
              <div class="target-head">
                <div>
                  <div class="k">Current target</div>
                  <div id="targetLetter" class="target-letter">L</div>
                </div>
                <div id="targetStep" class="target-step">Step 1 / 5</div>
              </div>
              <img id="targetImage" alt="ASL target handshape" />
              <div id="targetHint" class="hint"></div>
              <div class="source-link">
                Visual reference source:
                <a href="https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet.png" target="_blank" rel="noopener noreferrer">Wikimedia Commons ASL alphabet chart</a>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>

  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
  <script>
    const banner = document.getElementById("banner");
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const snapshotCanvas = document.getElementById("snapshotCanvas");
    const placeholder = document.getElementById("placeholder");
    const snapshot = document.getElementById("snapshot");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const captureButton = document.getElementById("captureButton");
    const captureLetterButton = document.getElementById("captureLetterButton");
    const resetDemoButton = document.getElementById("resetDemoButton");
    const cameraAccess = document.getElementById("cameraAccess");
    const streamState = document.getElementById("streamState");
    const capturedFrames = document.getElementById("capturedFrames");
    const handDetected = document.getElementById("handDetected");
    const handedness = document.getElementById("handedness");
    const landmarkCount = document.getElementById("landmarkCount");
    const topPrediction = document.getElementById("topPrediction");
    const lastAction = document.getElementById("lastAction");
    const rawPrediction = document.getElementById("rawPrediction");
    const top3 = document.getElementById("top3");
    const capturedLetters = document.getElementById("capturedLetters");
    const words = document.getElementById("words");
    const demoWordEl = document.getElementById("demoWord");
    const targetLetterEl = document.getElementById("targetLetter");
    const targetStepEl = document.getElementById("targetStep");
    const targetImageEl = document.getElementById("targetImage");
    const targetHintEl = document.getElementById("targetHint");
    const cameraTargetLetterEl = document.getElementById("cameraTargetLetter");
    const cameraTargetStepEl = document.getElementById("cameraTargetStep");
    const cameraTargetImageEl = document.getElementById("cameraTargetImage");
    const cameraTargetHintEl = document.getElementById("cameraTargetHint");

    let stream = null;
    let capturedCount = 0;
    let hands = null;
    let animationFrameId = null;
    let lastVideoTime = -1;
    let inferInFlight = false;
    let frameCounter = 0;
    let latestPrediction = "";
    let committedText = "";
    const landmarkBuffer = [];
    const MAX_BUFFER = 160;
    const DEMO_WORD = [
      {
        letter: "L",
        image: "https://upload.wikimedia.org/wikipedia/commons/0/08/Asl_alphabet_gallaudet.png",
        hint: "Target L. Open thumb and index finger to form an L-shape. Keep the other fingers folded. Look at the alphabet chart and compare your hand with the L cell."
      },
      {
        letter: "A",
        image: "https://upload.wikimedia.org/wikipedia/commons/0/08/Asl_alphabet_gallaudet.png",
        hint: "Target A. Make a fist with the thumb resting along the side of the hand."
      },
      {
        letter: "U",
        image: "https://upload.wikimedia.org/wikipedia/commons/0/08/Asl_alphabet_gallaudet.png",
        hint: "Target U. Raise the index and middle fingers together, parallel and close. Fold the rest."
      },
      {
        letter: "R",
        image: "https://upload.wikimedia.org/wikipedia/commons/0/08/Asl_alphabet_gallaudet.png",
        hint: "Target R. Cross the index and middle fingers. Fold the other fingers down."
      },
      {
        letter: "A",
        image: "https://upload.wikimedia.org/wikipedia/commons/0/08/Asl_alphabet_gallaudet.png",
        hint: "Repeat A. Fist closed, thumb resting to the side."
      }
    ];
    const DEMO_STABILITY_REQUIRED = 3;
    let demoIndex = 0;
    let demoStableCount = 0;
    let latestTop3 = [];
    let latestRawPrediction = "";
    const overlayCtx = overlay.getContext("2d");

    function setBanner(message, visible = true) {
      banner.textContent = message;
      banner.style.display = visible ? "block" : "none";
    }

    function renderDemo() {
      demoWordEl.innerHTML = DEMO_WORD.map((item, index) => {
        const state =
          index < demoIndex ? "done" :
          index === demoIndex ? "active" :
          "";
        return `<div class="demo-letter ${state}">${item.letter}</div>`;
      }).join("");

      const current = DEMO_WORD[Math.min(demoIndex, DEMO_WORD.length - 1)];
      if (!current) return;
      targetLetterEl.textContent = demoIndex >= DEMO_WORD.length ? "DONE" : current.letter;
      targetStepEl.textContent = demoIndex >= DEMO_WORD.length ? "Completed" : `Step ${demoIndex + 1} / ${DEMO_WORD.length}`;
      targetImageEl.src = current.image;
      targetHintEl.textContent = current.hint;
      cameraTargetLetterEl.textContent = targetLetterEl.textContent;
      cameraTargetStepEl.textContent = targetStepEl.textContent;
      cameraTargetImageEl.src = current.image;
      cameraTargetHintEl.textContent = current.hint;
    }

    function resetDemoState() {
      demoIndex = 0;
      demoStableCount = 0;
      committedText = "";
      latestPrediction = "";
      capturedLetters.textContent = "-";
      words.textContent = "-";
      renderDemo();
    }

    function processDemoPrediction(prediction) {
      if (demoIndex >= DEMO_WORD.length) return;
      const target = DEMO_WORD[demoIndex].letter.toLowerCase();
      const current = (prediction || "").toLowerCase();
      const raw = (latestRawPrediction || "").toLowerCase();
      const inTop3 = (latestTop3 || []).some((item) => String(item.char || "").toLowerCase() === target && Number(item.prob || 0) >= 0.12);
      const matched = current === target || raw.includes(target) || inTop3;
      if (!current && !raw && !inTop3) {
        demoStableCount = 0;
        return;
      }
      if (matched) {
        demoStableCount += 1;
        if (demoStableCount >= DEMO_STABILITY_REQUIRED) {
          committedText += DEMO_WORD[demoIndex].letter;
          capturedLetters.textContent = committedText;
          words.textContent = committedText;
          demoIndex += 1;
          demoStableCount = 0;
          if (demoIndex >= DEMO_WORD.length) {
            lastAction.textContent = "Demo completed";
            setBanner("Demo completed: LAURA");
          } else {
            lastAction.textContent = `Target solved: ${target.toUpperCase()}`;
            setBanner(`Correct letter detected: ${target.toUpperCase()}. Move to the next target.`);
          }
          renderDemo();
        }
      } else {
        demoStableCount = 0;
      }
    }

    function setStreamUi(active) {
      startButton.disabled = active;
      stopButton.disabled = !active;
      captureButton.disabled = !active;
      captureLetterButton.disabled = !active;
      placeholder.style.display = active ? "none" : "block";
      video.style.display = active ? "block" : "none";
      streamState.textContent = active ? "Streaming" : "Idle";
      if (!active) {
        overlayCtx.clearRect(0, 0, overlay.width || 1, overlay.height || 1);
      }
    }

    function resetHandUi() {
      handDetected.textContent = "No";
      handDetected.className = "v pill bad";
      handedness.textContent = "-";
      landmarkCount.textContent = "0";
      topPrediction.textContent = "-";
      rawPrediction.textContent = "-";
      top3.textContent = "-";
    }

    function ensureOverlaySize() {
      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      if (overlay.width !== width || overlay.height !== height) {
        overlay.width = width;
        overlay.height = height;
      }
    }

    function onHandsResults(results) {
      ensureOverlaySize();
      overlayCtx.save();
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      overlayCtx.drawImage(results.image, 0, 0, overlay.width, overlay.height);

      const landmarks = results.multiHandLandmarks || [];
      const handed = results.multiHandedness || [];
      if (landmarks.length > 0) {
        handDetected.textContent = "Yes";
        handDetected.className = "v pill good";
        handedness.textContent = handed[0] && handed[0].label ? handed[0].label : "-";
        landmarkCount.textContent = String(landmarks[0].length || 0);
        lastAction.textContent = "Hand tracked";
        const firstHand = landmarks[0];
        const handedLabel = handed[0] && handed[0].label ? handed[0].label : "";
        const xs = firstHand.map((lm) => handedLabel === "Left" ? (1 - lm.x) : lm.x);
        const ys = firstHand.map((lm) => lm.y);
        const zs = firstHand.map((lm) => lm.z);
        const frame63 = xs.concat(ys).concat(zs);
        landmarkBuffer.push(frame63);
        if (landmarkBuffer.length > MAX_BUFFER) {
          landmarkBuffer.shift();
        }
        frameCounter += 1;
        if (frameCounter % 6 === 0) {
          runInference(handedLabel);
        }
        for (const handLandmarks of landmarks) {
          drawConnectors(overlayCtx, handLandmarks, HAND_CONNECTIONS, {
            color: "#16a34a",
            lineWidth: 4
          });
          drawLandmarks(overlayCtx, handLandmarks, {
            color: "#f97316",
            lineWidth: 2,
            radius: 4
          });
        }
      } else {
        resetHandUi();
      }
      overlayCtx.restore();
    }

    async function runInference(handedLabel) {
      if (inferInFlight || landmarkBuffer.length < 12) return;
      inferInFlight = true;
      try {
        const response = await fetch("/webcam/infer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            frames: landmarkBuffer,
            handedness: handedLabel || null,
            min_frames: 12
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.error || "Inference failed");
        }
        topPrediction.textContent = data.top_prediction || "-";
        rawPrediction.textContent = data.raw_prediction || "-";
        top3.textContent = (data.top3 || []).map((item) => `${item.char}:${item.prob.toFixed(2)}`).join(" | ") || "-";
        latestPrediction = data.top_prediction || "";
        latestRawPrediction = data.raw_prediction || "";
        latestTop3 = data.top3 || [];
        processDemoPrediction(latestPrediction);
      } catch (error) {
        setBanner("Inference error: " + error.message);
      } finally {
        inferInFlight = false;
      }
    }

    async function initHands() {
      if (hands) return;
      hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      hands.onResults(onHandsResults);
    }

    async function processFrame() {
      if (!stream || !hands) return;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        await hands.send({ image: video });
      }
      animationFrameId = window.requestAnimationFrame(processFrame);
    }

    async function startWebcam() {
      try {
        setBanner("Requesting browser webcam access...");
        await initHands();
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
        cameraAccess.textContent = "Granted";
        cameraAccess.className = "v pill good";
        lastAction.textContent = "Camera started";
        setStreamUi(true);
        resetHandUi();
        landmarkBuffer.length = 0;
        resetDemoState();
        latestTop3 = [];
        latestRawPrediction = "";
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
        }
        animationFrameId = window.requestAnimationFrame(processFrame);
        setBanner("Webcam active. Browser hand tracking is running.");
      } catch (error) {
        cameraAccess.textContent = "Denied";
        cameraAccess.className = "v pill bad";
        lastAction.textContent = "Access failed";
        setBanner("Webcam error: " + error.message);
      }
    }

    function stopWebcam() {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      video.srcObject = null;
      setStreamUi(false);
      resetHandUi();
      landmarkBuffer.length = 0;
      resetDemoState();
      latestTop3 = [];
      latestRawPrediction = "";
      lastAction.textContent = "Camera stopped";
      setBanner("Webcam stopped.");
    }

    function captureFrame() {
      if (!stream) {
        setBanner("Start the webcam first.");
        return;
      }
      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      snapshotCanvas.width = width;
      snapshotCanvas.height = height;
      const ctx = snapshotCanvas.getContext("2d");
      ctx.drawImage(video, 0, 0, width, height);
      snapshot.src = snapshotCanvas.toDataURL("image/png");
      snapshot.style.display = "block";
      capturedCount += 1;
      capturedFrames.textContent = String(capturedCount);
      lastAction.textContent = "Frame captured";
      setBanner("Frame captured locally in the browser with landmark overlay available on screen.");
    }

    function capturePrediction() {
      if (!latestPrediction) {
        setBanner("There is no live prediction to capture yet.");
        return;
      }
      committedText += latestPrediction;
      capturedLetters.textContent = committedText || "-";
      words.textContent = committedText || "-";
      lastAction.textContent = `Captured '${latestPrediction}'`;
      setBanner(`Captured prediction: ${latestPrediction}`);
    }

    startButton.addEventListener("click", startWebcam);
    stopButton.addEventListener("click", stopWebcam);
    captureButton.addEventListener("click", captureFrame);
    captureLetterButton.addEventListener("click", capturePrediction);
    resetDemoButton.addEventListener("click", () => {
      resetDemoState();
      setBanner("Demo reset. Start again from L.");
    });
    setStreamUi(false);
    resetHandUi();
    renderDemo();
  </script>
</body>
</html>
    """


@app.post("/webcam/infer")
def webcam_infer(request: WebcamInferRequest) -> dict:
    runtime = _get_cached_webcam_runtime(request.ckpt)
    frames = [frame for frame in request.frames if len(frame) == 63]
    if len(frames) < int(request.min_frames):
        return {
            "status": "ok",
            "top_prediction": "",
            "raw_prediction": "",
            "top3": [],
            "frames_used": len(frames),
        }

    history = []
    for frame in frames[-runtime["max_frames"]:]:
        arr = torch.tensor(frame, dtype=torch.float32).numpy()
        history.append(arr)

    x_np, valid_t = build_model_input_from_history(
        history_raw63=history,
        expected_dim=runtime["expected_dim"],
        max_frames=runtime["max_frames"],
        normalize_landmarks=runtime["normalize_landmarks"],
        use_delta_features=runtime["use_delta_features"],
    )
    if valid_t <= 0:
        return {
            "status": "ok",
            "top_prediction": "",
            "raw_prediction": "",
            "top3": [],
            "frames_used": 0,
        }

    x = torch.from_numpy(x_np).unsqueeze(0).to(runtime["device"])
    input_lens = torch.tensor([valid_t], dtype=torch.long, device=runtime["device"])

    with torch.no_grad():
        try:
            log_probs = runtime["model"](x, input_lens)
        except TypeError:
            log_probs = runtime["model"](x)
        valid_log_probs = log_probs[:valid_t, :, :] if valid_t > 0 else log_probs[:1, :, :]
        probs = torch.exp(valid_log_probs[:, 0, :])
        mean_probs = probs.mean(dim=0)

    nonblank = mean_probs.clone()
    blank_id = runtime["blank_id"]
    if 0 <= blank_id < nonblank.shape[0]:
        nonblank[blank_id] = 0.0
    vals, idxs = torch.topk(nonblank, k=min(3, nonblank.shape[0]))
    top3 = [
        {
            "char": runtime["idx2char"].get(int(i), f"#{int(i)}"),
            "prob": float(v),
        }
        for i, v in zip(idxs.detach().cpu().tolist(), vals.detach().cpu().tolist())
    ]
    top_prediction = sanitize_decoded_text(
        top3[0]["char"] if top3 else "",
        runtime["lowercase"],
        runtime["letters_only"],
    )
    raw_prediction = sanitize_decoded_text(
        ctc_decode_text(valid_log_probs, runtime["idx2char"], blank_id),
        runtime["lowercase"],
        runtime["letters_only"],
    )
    return {
        "status": "ok",
        "frames_used": int(valid_t),
        "top_prediction": top_prediction,
        "raw_prediction": raw_prediction,
        "top3": top3,
    }


@app.post("/offline-test/run")
def run_offline_test(request: OfflineTestRequest) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("artifacts/eval/web_runs") / f"{request.mode}_{timestamp}"
    max_eval_rows, batch_size = _resolve_request_settings(request)

    if request.mode == "quick":
        return _run_pr3_quick_eval(request)
    else:
        artifacts = run_test(
            ckpt_path=Path(request.ckpt),
            data_dir=Path(request.data_dir),
            hand_model_path=Path(request.hand_model),
            output_dir=output_dir,
            val_ratio=float(request.val_ratio),
            seed=int(request.seed),
            batch_size=batch_size,
            run_webcam_check=False,
            camera_index=0,
            split_csv=Path(request.split_csv) if request.split_csv else None,
            max_eval_rows=max_eval_rows,
        )
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        artifacts_dict = {
            "summary": str(artifacts.summary_path),
            "cases": str(artifacts.cases_path),
        }

    return {
        "status": "ok",
        "message": f'Offline evaluation completed with mode="{request.mode}" and maxSamples={max_eval_rows}',
        "result": _summary_to_web_json(summary, mode=request.mode, max_samples=max_eval_rows),
        "artifacts": artifacts_dict,
    }
