from hashlib import md5
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

wav_folders = ["/mnt/nvme0/wenet-speech/sliced-m", "/mnt/nvme0/libri-speech"]

wavs: list[Path] = []
for i in wav_folders:
    wavs.extend(list(Path(i).rglob("*.flac")) + list(Path(i).rglob("*.wav")))

wavs.sort()

print("Total wavs:", len(wavs))

model_path = "TencentGameMate/chinese-hubert-base"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = HubertModel.from_pretrained(model_path)

device = "cuda"

model = model.to(device)
model = model.half()
model.eval()

metadata = open("/mnt/nvme0/hubert-features/metadata.csv", "w")

for wav_path in tqdm(wavs):
    md5_hash = md5(wav_path.read_bytes()).hexdigest()

    wav, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    input_values = feature_extractor(
        wav, return_tensors="pt", sampling_rate=sr
    ).input_values
    input_values = input_values.half()
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state

    saved_path = f"/mnt/nvme0/hubert-features/{md5_hash}.pt"
    torch.save(last_hidden_state[0], saved_path)

    metadata.write(f"{wav_path}\t{saved_path}\n")

metadata.close()
