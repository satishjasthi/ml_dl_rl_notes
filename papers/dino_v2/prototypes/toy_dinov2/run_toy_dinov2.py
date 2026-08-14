"""Tiny CPU-friendly DINOv2 teaching prototype.

The implementation intentionally exposes the three learning signals:
DINO-style global distillation, iBOT-style masked patch prediction, and a
KoLeo-style feature-spacing regularizer.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class BatchViews:
    image: Tensor
    global_views: list[Tensor]
    local_views: list[Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_images(num_images: int, size: int = 16) -> Tensor:
    """Create repeated, unlabeled visual patterns with small noise."""
    images = torch.zeros(num_images, 1, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for index in range(num_images):
        kind = index % 3
        if kind == 0:
            pattern = ((xx // 2) % 2).float()
        elif kind == 1:
            pattern = ((yy // 2) % 2).float()
        else:
            center = (size - 1) / 2
            radius = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)
            pattern = (radius < size * 0.3).float()
        images[index, 0] = pattern + 0.05 * torch.randn(size, size)
    return images.clamp(0.0, 1.0)


def crop_and_resize(image: Tensor, crop_size: int, generator: torch.Generator) -> Tensor:
    """Random crop followed by resize back to the encoder's image size."""
    _, height, width = image.shape
    top = int(torch.randint(0, height - crop_size + 1, (1,), generator=generator))
    left = int(torch.randint(0, width - crop_size + 1, (1,), generator=generator))
    crop = image[:, top : top + crop_size, left : left + crop_size]
    return F.interpolate(crop.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)


def make_views(images: Tensor, seed: int) -> BatchViews:
    generator = torch.Generator().manual_seed(seed)
    global_views = [
        torch.stack([crop_and_resize(image, 12, generator) for image in images]),
        torch.stack([crop_and_resize(image, 12, generator) for image in images]),
    ]
    local_views = [
        torch.stack([crop_and_resize(image, 8, generator) for image in images]),
        torch.stack([crop_and_resize(image, 8, generator) for image in images]),
    ]
    return BatchViews(images, global_views, local_views)


class TinyPatchEncoder(nn.Module):
    def __init__(self, image_size: int = 16, patch_size: int = 4, dim: int = 48) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, self.num_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, images: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        if patch_mask is not None:
            tokens = tokens.masked_fill(patch_mask.unsqueeze(-1), 0.0)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1) + self.position
        encoded = self.norm(self.transformer(sequence))
        return encoded[:, 0], encoded[:, 1:]


class TinyDinoHead(nn.Module):
    def __init__(self, dim: int, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, output_dim))

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class ToyDinoV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyPatchEncoder()
        self.head = TinyDinoHead(48)

    def forward(
        self, images: Tensor, patch_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        cls, patches = self.encoder(images, patch_mask)
        return self.head(cls), self.head(patches), cls


def cross_entropy_from_teacher(teacher_logits: Tensor, student_logits: Tensor, temperature: float) -> Tensor:
    teacher_probs = teacher_logits.detach()
    return -(teacher_probs * F.log_softmax(student_logits / temperature, dim=-1)).sum(dim=-1).mean()


def koleo_loss(features: Tensor) -> Tensor:
    normalized = F.normalize(features, dim=-1)
    distances = torch.cdist(normalized, normalized)
    diagonal = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    distances = distances.masked_fill(diagonal, float("inf"))
    nearest = distances.min(dim=1).values
    return -torch.log(nearest.clamp_min(1e-6)).mean()


def teacher_distribution(logits: Tensor, center: Tensor, temperature: float) -> Tensor:
    return F.softmax((logits - center) / temperature, dim=-1)


class ToyTrainer:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.student = ToyDinoV2().to(self.device)
        self.teacher = copy.deepcopy(self.student).to(self.device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=2e-3, weight_decay=1e-4)
        self.center = torch.zeros(1, 32, device=self.device)

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.99) -> None:
        for teacher_parameter, student_parameter in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1.0 - momentum)

    def step(self, views: BatchViews, seed: int) -> dict[str, float]:
        global_a, global_b = (view.to(self.device) for view in views.global_views)
        local_a, local_b = (view.to(self.device) for view in views.local_views)
        with torch.no_grad():
            teacher_a, teacher_patches_a, _ = self.teacher(global_a)
            teacher_b, teacher_patches_b, _ = self.teacher(global_b)
            teacher_a = teacher_distribution(teacher_a, self.center, 0.04)
            teacher_b = teacher_distribution(teacher_b, self.center, 0.04)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        patch_mask = torch.rand(
            global_a.shape[0], self.student.encoder.num_patches, generator=generator, device=self.device
        ) < 0.5
        student_global_a, student_patches_a, student_features = self.student(global_a, patch_mask)
        student_global_b, _, _ = self.student(global_b)
        student_local_a, _, _ = self.student(local_a)
        student_local_b, _, _ = self.student(local_b)
        global_loss = sum(
            cross_entropy_from_teacher(teacher_target, student_logits, 0.1)
            for teacher_target in (teacher_a, teacher_b)
            for student_logits in (student_global_a, student_global_b, student_local_a, student_local_b)
        ) / 8.0
        teacher_patch_target = teacher_patches_a.detach()[patch_mask]
        student_patch_logits = student_patches_a
        patch_loss = cross_entropy_from_teacher(
            teacher_distribution(teacher_patch_target, torch.zeros_like(self.center), 0.04),
            student_patch_logits[patch_mask],
            0.1,
        )
        feature_loss = koleo_loss(student_features)
        total_loss = global_loss + 0.5 * patch_loss + 0.05 * feature_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            batch_center = torch.cat([self.teacher.head(self.teacher.encoder(global_a)[0]), self.teacher.head(self.teacher.encoder(global_b)[0])]).mean(0, keepdim=True)
            self.center.mul_(0.9).add_(batch_center, alpha=0.1)
            self.update_teacher()
        return {
            "global": float(global_loss.detach()),
            "patch": float(patch_loss.detach()),
            "koleo": float(feature_loss.detach()),
            "total": float(total_loss.detach()),
        }


def run(epochs: int = 8, batch_size: int = 12, seed: int = 7) -> dict[str, float]:
    set_seed(seed)
    images = make_synthetic_images(48)
    trainer = ToyTrainer()
    last: dict[str, float] = {}
    for epoch in range(epochs):
        permutation = torch.randperm(len(images))
        metrics = []
        for batch_index, indices in enumerate(permutation.split(batch_size)):
            batch = images[indices]
            views = make_views(batch, seed + epoch * 100 + batch_index)
            metrics.append(trainer.step(views, seed + epoch * 1000 + batch_index))
        last = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
        print(
            f"epoch={epoch + 1:02d} global={last['global']:.4f} "
            f"patch={last['patch']:.4f} koleo={last['koleo']:.4f} total={last['total']:.4f}"
        )
    with torch.no_grad():
        features = trainer.student(images)[2]
    last["feature_std"] = float(features.std())
    last["teacher_student_distance"] = float(
        torch.sqrt(sum((a - b).pow(2).sum() for a, b in zip(trainer.student.parameters(), trainer.teacher.parameters())))
    )
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    metrics = run(args.epochs, args.batch_size, args.seed)
    print(f"feature_std={metrics['feature_std']:.6f}")
    print(f"teacher_student_distance={metrics['teacher_student_distance']:.6f}")


if __name__ == "__main__":
    main()
