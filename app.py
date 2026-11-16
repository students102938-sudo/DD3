import io
import os
import wave
import math
import struct
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload


def detect_bpm_from_samples(samples: np.ndarray, sample_rate: int) -> float:
    """
    Python-side Onset Energy Envelope + Autocorrelation BPM Detector
    """
    if len(samples) < sample_rate * 2:
        return 120.0

    # Mono mix if stereo
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    # Downsample envelope calculation using 10ms hops
    hop_size = int(sample_rate * 0.01)
    num_frames = len(samples) // hop_size
    if num_frames < 100:
        return 120.0

    frames = samples[:num_frames * hop_size].reshape(num_frames, hop_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Half-wave rectified spectral/energy flux (onset strength)
    onset_env = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    onset_env -= np.mean(onset_env)

    # Autocorrelation over BPM range 60 to 180 BPM
    env_rate = sample_rate / hop_size  # ~100 Hz
    min_lag = int(env_rate * 60.0 / 180.0)
    max_lag = int(env_rate * 60.0 / 60.0)

    corr = np.correlate(onset_env, onset_env, mode='full')
    corr = corr[len(corr) // 2:]

    if max_lag >= len(corr):
        return 120.0

    search_region = corr[min_lag:max_lag + 1]
    best_lag = min_lag + int(np.argmax(search_region))
    bpm = 60.0 * env_rate / best_lag

    # Normalize into musical 75 - 155 BPM range
    while bpm < 75.0:
        bpm *= 2.0
    while bpm > 155.0:
        bpm /= 2.0

    return round(float(bpm), 1)


def apply_dd3_dsp_python(samples: np.ndarray, sample_rate: int, delay_ms: float, feedback: float, e_level: float) -> np.ndarray:
    """
    Authentic BOSS DD-3 Digital Delay DSP Simulation in Python (NumPy):
    - Circular Ring Buffer (12.5ms - 800ms)
    - 12-bit DAC/ADC quantization character
    - 1-Pole 7kHz Analog Low-Pass Filter inside the feedback loop
    """
    is_stereo = samples.ndim > 1
    if not is_stereo:
        samples = samples[:, np.newaxis]

    num_samples, num_channels = samples.shape
    delay_samples = max(1, int(sample_rate * (delay_ms / 1000.0)))

    # 1-pole low-pass filter coefficient for ~7kHz cutoff
    cutoff_hz = 7000.0
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    # Ring buffer
    buffer_len = delay_samples + 2
    ring_buffer = np.zeros((buffer_len, num_channels), dtype=np.float32)
    lp_state = np.zeros(num_channels, dtype=np.float32)

    output = np.zeros_like(samples, dtype=np.float32)
    write_idx = 0

    fb = min(0.95, max(0.0, feedback))
    wet = min(1.0, max(0.0, e_level))

    # 12-bit quantization step (4096 levels)
    quant_levels = 2048.0

    for n in range(num_samples):
        read_idx = (write_idx - delay_samples) % buffer_len
        delayed = ring_buffer[read_idx]

        # 12-bit quantization simulation on delayed signal
        delayed_12bit = np.round(delayed * quant_levels) / quant_levels

        # Analog 7kHz low-pass filter on feedback path
        lp_state = lp_state + alpha * (delayed_12bit - lp_state)

        dry_in = samples[n]
        ring_buffer[write_idx] = dry_in + lp_state * fb

        # Mix dry + wet
        mixed = dry_in + (lp_state * wet)
        output[n] = np.clip(mixed, -1.0, 1.0)

        write_idx = (write_idx + 1) % buffer_len

    if not is_stereo:
        return output[:, 0]
    return output


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "engine": "BOSS DD-3 Python DSP + WebAudio"})


@app.route('/api/analyze-wav', methods=['POST'])
def analyze_wav():
    """Analyze uploaded WAV file in Python to detect exact BPM."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    try:
        raw_bytes = f.read()
        with wave.open(io.BytesIO(raw_bytes), 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            return jsonify({"bpm": 120.0, "note": "Unsupported bit depth for server analysis, using client BPM"})

        if channels > 1:
            data = data.reshape(-1, channels)

        bpm = detect_bpm_from_samples(data, sr)
        return jsonify({"bpm": bpm, "sample_rate": sr, "channels": channels})
    except Exception as e:
        return jsonify({"bpm": 120.0, "warning": str(e)})


@app.route('/api/export-wav', methods=['POST'])
def export_wav():
    """Process a WAV file through Python DD-3 DSP engine and return downloadable delayed WAV."""
    if 'file' not in request.files:
        return jsonify({"error": "No WAV file provided"}), 400

    f = request.files['file']
    delay_ms = float(request.form.get('delay_ms', 375.0))
    feedback = float(request.form.get('feedback', 0.45))
    e_level = float(request.form.get('e_level', 0.65))

    try:
        raw_bytes = f.read()
        with wave.open(io.BytesIO(raw_bytes), 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth != 2:
            return jsonify({"error": "Only 16-bit PCM WAV files are supported for server export."}), 400

        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels)

        processed = apply_dd3_dsp_python(data, sr, delay_ms, feedback, e_level)
        out_int16 = np.int16(np.clip(processed * 32767.0, -32768, 32767))

        out_buf = io.BytesIO()
        with wave.open(out_buf, 'wb') as wf_out:
            wf_out.setnchannels(channels)
            wf_out.setsampwidth(2)
            wf_out.setframerate(sr)
            wf_out.writeframes(out_int16.tobytes())

        out_buf.seek(0)
        return send_file(
            out_buf,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='dd3_delayed_output.wav'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
