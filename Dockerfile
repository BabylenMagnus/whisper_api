FROM huggingface/transformers-pytorch-gpu

RUN pip install Flask[async]
RUN pip install numpy
RUN pip install torch torchvision
RUN pip install tqdm
RUN pip install transformers
RUN pip install ffmpeg-python
RUN apt install ffmpeg -y
RUN pip install soundfile
RUN pip install scipy
RUN pip install pybase64
RUN pip install faster-whisper
RUN pip install yt-dlp

COPY whisper whisper/
WORKDIR whisper/
EXPOSE 5000
CMD python3 app.py
