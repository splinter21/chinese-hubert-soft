import librosa
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor


class KMeansDataset(Dataset):
    def __init__(self):
        self.dataset = open("/mnt/nvme0/hubert-features/metadata.csv").readlines()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "TencentGameMate/chinese-hubert-base"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_path, feature_path = self.dataset[idx].strip().split("\t")

        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        input_values = self.feature_extractor(
            wav, return_tensors="pt", sampling_rate=sr
        ).input_values[0]

        input_values = input_values[: 16000 * 20]

        return dict(
            input_values=input_values,
        )


if __name__ == "__main__":
    dataset = KMeansDataset()

    print(dataset[0])
