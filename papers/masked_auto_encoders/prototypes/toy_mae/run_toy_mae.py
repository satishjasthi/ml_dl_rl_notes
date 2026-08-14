"""Small educational masked autoencoder demonstration."""

from __future__ import annotations

import argparse
import random

import torch
from torch import Tensor, nn


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        image_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 1,
        embed_dim: int = 32,
        decoder_dim: int = 16,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.encoder_pos = nn.Parameter(torch.zeros(self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(self.num_patches, decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=4,
            dim_feedforward=decoder_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.encoder_pos, std=0.02)
        nn.init.normal_(self.decoder_pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, images: Tensor) -> Tensor:
        batch, channels, height, width = images.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError("images must have the configured square image size")
        p = self.patch_size
        patches = images.reshape(batch, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, self.num_patches, self.patch_dim)

    def random_mask(self, batch_size: int, mask_ratio: float, device: torch.device) -> tuple[Tensor, Tensor]:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        keep = int(self.num_patches * (1.0 - mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_keep = noise.argsort(dim=1)[:, :keep]
        mask = torch.ones(batch_size, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_keep, False)
        return ids_keep, mask

    def encode(self, images: Tensor) -> Tensor:
        patches = self.patchify(images)
        tokens = self.patch_embed(patches) + self.encoder_pos.unsqueeze(0)
        return self.encoder(tokens)

    def forward(self, images: Tensor, mask_ratio: float = 0.75) -> tuple[Tensor, Tensor, Tensor]:
        patches = self.patchify(images)
        ids_keep, mask = self.random_mask(images.shape[0], mask_ratio, images.device)
        visible = patches.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, self.patch_dim))
        visible = self.patch_embed(visible)
        visible = visible + self.encoder_pos[ids_keep]
        encoded = self.encoder(visible)

        decoded_visible = self.decoder_embed(encoded)
        decoded = self.mask_token.expand(images.shape[0], self.num_patches, -1).clone()
        decoded.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, decoded_visible.shape[-1]), decoded_visible)
        decoded = self.decoder(decoded + self.decoder_pos.unsqueeze(0))
        prediction = self.decoder_pred(decoded)

        target = (patches - patches.mean(dim=-1, keepdim=True)) / (patches.std(dim=-1, keepdim=True) + 1e-6)
        per_patch_loss = (prediction - target).pow(2).mean(dim=-1)
        loss = (per_patch_loss * mask.float()).sum() / mask.sum().clamp_min(1)
        return loss, prediction, mask


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


def train_demo(seed: int = 7, pretrain_epochs: int = 12, classifier_epochs: int = 20) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    images, labels = make_dataset(128, seed=seed)
    train_images, test_images = images[:96], images[96:]
    train_labels, test_labels = labels[:96], labels[96:]

    model = MaskedAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    last_loss = 0.0
    for _ in range(pretrain_epochs):
        optimizer.zero_grad()
        last_loss, _, _ = model(train_images)
        last_loss.backward()
        optimizer.step()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    classifier = nn.Linear(32, 2)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
    for _ in range(classifier_epochs):
        classifier_optimizer.zero_grad()
        logits = classifier(model.encode(train_images).mean(dim=1))
        loss = nn.functional.cross_entropy(logits, train_labels)
        loss.backward()
        classifier_optimizer.step()

    with torch.no_grad():
        test_logits = classifier(model.encode(test_images).mean(dim=1))
        accuracy = (test_logits.argmax(dim=1) == test_labels).float().mean().item()
    return {"reconstruction_loss": float(last_loss.item()), "linear_accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    args = parser.parse_args()
    metrics = train_demo(args.seed, args.pretrain_epochs, args.classifier_epochs)
    print(f"masked-patch reconstruction loss: {metrics['reconstruction_loss']:.4f}")
    print(f"frozen-encoder linear accuracy: {metrics['linear_accuracy']:.3f}")


if __name__ == "__main__":
    main()
