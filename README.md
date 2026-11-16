# BOSS DD-3 Digital Delay — Auto-Tempo Web App (Python + Flask)

A full-stack Python web application emulating the iconic **BOSS DD-3 Digital Delay** guitar/vocal effects pedal with **Real-Time Auto-Tempo BPM Locking** and a **Python NumPy DSP Engine**.

## Features
- **Real-Time Auto-Tempo Engine:** Automatically detects incoming BPM from uploaded audio files, live microphone/guitar input, or the built-in synth groove, and locks the delay time (`D.TIME`) to your chosen rhythmic subdivision (`1/4`, `Dotted 1/8`, `1/8`, `1/8 Triplet`).
- **Authentic BOSS DD-3 Controls:** Interactive rotary knobs for `E.LEVEL`, `F.BACK`, `D.TIME`, and `MODE` (`50ms`, `200ms`, `800ms`, `HOLD`), plus footswitch bypass and LED indicator.
- **Python DSP Backend (`app.py`):**
  - `/api/analyze-wav`: Autocorrelation onset energy BPM detector in NumPy.
  - `/api/export-wav`: Offline 12-bit circular ring-buffer delay + 7kHz 1-pole analog low-pass feedback filter to export processed `.wav` files.
- **Drag & Drop Audio Player:** Works smoothly when hosted locally or deployed online without sandbox iframe restrictions.

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000` in your browser.

## Deploy via GitHub (Render / Railway / Koyeb)
1. Push this folder to a new GitHub repository.
2. Connect the repository on **Render.com** (Blueprint / Web Service) or **Railway.app**.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
