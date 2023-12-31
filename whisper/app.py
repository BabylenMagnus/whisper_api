import io
import os

from flask import Flask, request, redirect, url_for

import soundfile as sf
import scipy.signal as sps

import numpy as np
import requests

import queue
import threading
from faster_whisper import WhisperModel

import yt_dlp as youtube_dl
import uuid

app = Flask(__name__, static_folder='results')

model_size = "large-v2"
model = WhisperModel(model_size, device="cuda", compute_type="int8")
SAMPLE_RATE = 16000


def queue_thread(load_queue, model_queue):
    while True:
        inp = load_queue.get()

        audio_ = inp.get('url')
        audio_ = requests.get(audio_)
        output, samplerate = sf.read(io.BytesIO(audio_.content))
        if samplerate != SAMPLE_RATE:
            number_of_samples = round(len(output) * float(SAMPLE_RATE) / samplerate)
            output = sps.resample(output, number_of_samples)
        output = np.array(output).astype(np.float32)
        model_queue.put((output, inp))


def youtube_download(load_queue, model_queue):
    while True:
        inp = load_queue.get()
        filename = "TEMP/" + str(uuid.uuid4()) + ".mp4"
        ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([inp["yt_url"]])
        inp["filename"] = filename

        model_queue.put((filename, inp))


def model_inference(model_queue, callback_queue):
    while True:
        audio, inp = model_queue.get()
        segments, _ = model.transcribe(audio, beam_size=5)
        segments = list(segments)

        if "filename" in inp:
            os.remove(inp["filename"])

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


@app.route('/audio_callback', methods=['POST'])
async def transcribe_audio():
    inp = request.get_json()
    load_queue.put(inp)
    return "1"


@app.route('/audio', methods=['POST'])
async def transcribe_audio():
    inp = request.get_json()
    audio_ = inp.get('url')
    audio_ = requests.get(audio_)
    output, samplerate = sf.read(io.BytesIO(audio_.content))
    if samplerate != SAMPLE_RATE:
        number_of_samples = round(len(output) * float(SAMPLE_RATE) / samplerate)
        output = sps.resample(output, number_of_samples)
    output = np.array(output).astype(np.float32)
    segments, _ = model.transcribe(output, beam_size=5)
    segments = list(segments)
    out = {
        "segments": segments
    }
    return out


@app.route('/youtube', methods=['POST'])
async def transcribe_youtube():
    inp = request.get_json()
    youtube_queue.put(inp)
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
youtube_queue = queue.Queue()
youtube_thread = threading.Thread(target=youtube_download, args=(youtube_queue, model_queue,), daemon=True)
youtube_thread.start()

if __name__ == '__main__':
    app.run(host="0.0.0.0")
