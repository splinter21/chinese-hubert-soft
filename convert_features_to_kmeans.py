import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

features_folder = Path("/mnt/nvme0/hubert-features/")
all_features = list(features_folder.rglob("*.pt"))

EPOCH = 20
BATCH_SIZE = 256 * 24

kmeans = MiniBatchKMeans(
    n_clusters=128, random_state=0, batch_size=BATCH_SIZE, n_init="auto"
)

for epoch in range(EPOCH):
    random.shuffle(all_features)

    for i in tqdm(range(0, len(all_features), BATCH_SIZE), desc=f"Epoch {epoch}"):
        batch_features = all_features[i : i + BATCH_SIZE]
        batch = torch.concat([torch.load(x) for x in batch_features], dim=0)
        kmeans.partial_fit(batch.cpu().numpy())


# Save the model
with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Save cluster centers
np.save("cluster_centers.npy", kmeans.cluster_centers_)
