import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader
from transformers import HubertModel

from dataset import KMeansDataset


class TeacherModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
        self.cluster_centers = nn.Parameter(
            torch.from_numpy(np.load("cluster_centers.npy")).float()
        )

    def forward(self, input_values, attention_mask=None):
        x = self.model(input_values, attention_mask=attention_mask)
        x = x.last_hidden_state

        # X in shape (batch_size, seq_len, 768)
        # cluster_centers in shape (128, 768)

        x = self.kmeans(x)

        return x

    def kmeans(self, x):
        distances = torch.cdist(x, self.cluster_centers)
        x = torch.argmin(distances, dim=-1)

        return x


class MyLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
        self.teacher = TeacherModel()
        self.teacher.freeze()

        embed_dim = 256
        self.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, embed_dim))
        self.label_embedding = nn.Embedding(128, embed_dim)
        self.loss = nn.CrossEntropyLoss()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, input_values, attention_mask=None):
        x = self.model(input_values, attention_mask=attention_mask)
        x = self.proj(x.last_hidden_state)

        return x

    def _step(self, batch, batch_idx, mode="train"):
        input_values = batch["input_values"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            labels = self.teacher(input_values, attention_mask=attention_mask)

        logits = self(input_values, attention_mask=attention_mask)
        logits = self.logits(logits)

        loss = self.loss(logits.flatten(0, 1), labels.flatten(0, 1))

        self.log(f"{mode}_loss", loss, prog_bar=False, sync_dist=True)

        x = logits.argmax(-1)
        avg_acc = (x == labels).float().mean()
        self.log(f"{mode}_acc", avg_acc, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(), lr=2e-5, weight_decay=1e-2, betas=(0.9, 0.98)
        )

        return optim


def collate(batch):
    # pad the inputs on the right up to the maximum length
    input_values = [i["input_values"] for i in batch]

    max_input_length = max([len(i) for i in input_values])
    input_values_mask = []
    for i in input_values:
        input_values_mask.append([1] * len(i) + [0] * (max_input_length - len(i)))

    input_values_padded = [
        torch.nn.functional.pad(i, (0, max_input_length - len(i)), value=0)
        for i in input_values
    ]

    return dict(
        input_values=torch.stack(input_values_padded),
        attention_mask=torch.tensor(input_values_mask),
    )


if __name__ == "__main__":
    pl.seed_everything(42)

    dataset = KMeansDataset()
    split = int(len(dataset) * 0.95)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split]
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=4, collate_fn=collate, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, collate_fn=collate, shuffle=True, num_workers=2
    )

    model = MyLightningModule()

    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=10,
        accumulate_grad_batches=16,
        val_check_interval=10000,
        check_val_every_n_epoch=None,
        max_epochs=100,
        precision=16,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_acc:.2f}",
                monitor="val_acc",
                mode="max",
                save_top_k=3,
                save_last=True,
            )
        ],
        logger=WandbLogger(
            project="hubert",
            save_dir="logs",
            log_model="all",
            entity="fish-audio",
            # resume="must", id="9srjca5y"
        ),
        # resume_from_checkpoint="logs/asr/9srjca5y/checkpoints/last.ckpt"
    )

    trainer.fit(model, train_loader, val_loader)
