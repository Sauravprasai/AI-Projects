from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = ("../../Models/models--openai--whisper-medium/"
              "snapshots/18530d7c5ce1083f21426064b85fbd1e24bd1858")

pipe = pipeline("automatic-speech-recognition",
                model=model_path,
                chunk_length_s=30,
                device=device)

output = pipe("../../AudioTest/Audio.mp3")["text"]
print(output)