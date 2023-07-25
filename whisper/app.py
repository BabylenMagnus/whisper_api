import io

from flask import Flask, jsonify, request, render_template, redirect, url_for

import soundfile as sf
import scipy.signal as sps

import numpy as np
import requests

import queue
import threading
from faster_whisper import WhisperModel


app = Flask(__name__, static_folder='results')

model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="int8")
SAMPLE_RATE = 16000


@app.route("/", methods=['GET', 'POST'])
def index():
    return redirect(url_for('upload_file'))


def queue_thread(load_queue, model_queue):
    while True:
        inp = load_queue.get()
        print("+-" * 20, "load")

        audio_ = inp.get('url')
        audio_ = requests.get(audio_)
        output, samplerate = sf.read(io.BytesIO(audio_.content))
        if samplerate != SAMPLE_RATE:
            number_of_samples = round(len(output) * float(SAMPLE_RATE) / samplerate)
            output = sps.resample(output, number_of_samples)
        output = np.array(output).astype(np.float32)
        model_queue.put((output, inp))


def model_inference(model_queue, callback_queue):
    while True:
        audio, inp = model_queue.get()
        segments, _ = model.transcribe(audio, beam_size=5)
        segments = list(segments)
        callback_queue.put((segments, inp))


def callback(q):
    while True:
        segments, inp = q.get()
        out = {
            "segments": segments
        }
        url = inp["callback_url"]
        out["request_hash"] = inp["request_hash"]
        requests.post(url, json=out)


@app.route('/audio', methods=['POST'])
async def transcribe_audio():
    inp = request.get_json()
    load_queue.put(inp)
    return "1"


callback_queue = queue.Queue()
callback_thread = threading.Thread(target=callback, args=(callback_queue,), daemon=True)
callback_thread.start()
model_queue = queue.Queue()
model_thread = threading.Thread(target=model_inference, args=(model_queue,callback_queue,), daemon=True)
model_thread.start()
load_queue = queue.Queue()
load_thread = threading.Thread(target=queue_thread, args=(load_queue, model_queue,), daemon=True)
load_thread.start()


if __name__ == '__main__':
    app.run(host="0.0.0.0")
