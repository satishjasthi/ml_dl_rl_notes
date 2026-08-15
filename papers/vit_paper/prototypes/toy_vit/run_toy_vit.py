"""Small educational Vision Transformer demonstration."""

from __future__ import annotations

import argparse
import random

import torch
from torch import Tensor, nn


class ToyViT(nn.Module):
    """A minimal patch-token Transformer for image classification."""

    def __init__(
        self,
        image_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 1,
        embed_dim: int = 32,
        num_heads: int = 4,
        depth: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embed, std=0.02)

    def patchify(self, images: Tensor) -> Tensor:
        batch, channels, height, width = images.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError("images must have the configured square image size")
        p = self.patch_size
        patches = images.reshape(batch, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, self.num_patches, self.patch_dim)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        patches = self.patchify(images)
        tokens = self.patch_embed(patches)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = self.encoder(tokens + self.position_embed)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0]), tokens


def make_dataset(num_samples: int, image_size: int = 8, seed: int = 7) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(num_samples, 1, image_size, image_size)
    labels = torch.arange(num_samples) % 2
    for index, label in enumerate(labels):
        if label == 0:
            images[index, 0, :, ::2] = 1.0
        else:
            images[index, 0, ::2, :] = 1.0
    images += 0.15 * torch.randn(images.shape, generator=generator)
    return images.clamp(0.0, 1.0), labels


def train_demo(seed: int = 7, epochs: int = 12, batch_size: int = 32) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    images, labels = make_dataset(128, seed=seed)
    train_images, test_images = images[:96], images[96:]
    train_labels, test_labels = labels[:96], labels[96:]

    model = ToyViT()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    last_loss = torch.tensor(float("nan"))
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(train_images.shape[0])
        for start in range(0, train_images.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            logits, _ = model(train_images[indices])
            last_loss = criterion(logits, train_labels[indices])
            last_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits, _ = model(test_images)
        accuracy = (test_logits.argmax(dim=1) == test_labels).float().mean().item()
    return {"loss": float(last_loss.item()), "accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    metrics = train_demo(args.seed, args.epochs, args.batch_size)
    print(f"final cross-entropy loss: {metrics['loss']:.4f}")
    print(f"synthetic stripe test accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
