"""Educational three-stage SimCLRv2 workflow on synthetic data.

This is a toy demonstration, not an ImageNet-scale reproduction:

    unlabeled views -> contrastive encoder -> few-label teacher -> student distillation

Run from the paper directory with:

    uv run --locked python prototypes/three_stage_toy/run_simclrv2_toy.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ToyData:
    points: Tensor
    labels: Tensor
    labeled_indices: Tensor
    unlabeled_indices: Tensor


class Encoder(nn.Module):
    def __init__(self, hidden: int = 32, representation_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, representation_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, representation_dim: int = 16) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(representation_dim, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))


def make_data(seed: int, points_per_class: int = 80) -> ToyData:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    chunks = [center + 0.55 * torch.randn(points_per_class, 2, generator=generator) for center in centers]
    points = torch.cat(chunks)
    labels = torch.arange(3).repeat_interleave(points_per_class)
    permutation = torch.randperm(points.shape[0], generator=generator)
    points, labels = points[permutation], labels[permutation]

    labeled_parts = [torch.where(labels == class_id)[0][:4] for class_id in range(3)]
    labeled_indices = torch.cat(labeled_parts)
    all_indices = torch.arange(points.shape[0])
    labeled_mask = torch.zeros(points.shape[0], dtype=torch.bool)
    labeled_mask[labeled_indices] = True
    return ToyData(points, labels, labeled_indices, all_indices[~labeled_mask])


def nt_xent_loss(z: Tensor, temperature: float = 0.5) -> Tensor:
    """Bidirectional in-batch contrastive loss for [view_1, view_2]."""
    if z.shape[0] % 2:
        raise ValueError("contrastive batch must contain an even number of views")
    n = z.shape[0] // 2
    normalized = F.normalize(z, dim=1)
    logits = normalized @ normalized.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits, targets)


def augment(points: Tensor, generator: torch.Generator) -> Tensor:
    """Create a stochastic view; noise stands in for image augmentations."""
    scale = 0.9 + 0.2 * torch.rand(points.shape[0], 1, generator=generator)
    noise = 0.12 * torch.randn(points.shape, generator=generator)
    return points * scale + noise


def pretrain_encoder(data: ToyData, seed: int, epochs: int = 80) -> tuple[Encoder, float]:
    encoder = Encoder()
    projector = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 1)
    last_loss = 0.0
    for _ in range(epochs):
        view_1 = augment(data.points, generator)
        view_2 = augment(data.points, generator)
        z = projector(torch.cat([encoder(view_1), encoder(view_2)]))
        loss = nt_xent_loss(z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return encoder, last_loss


def accuracy(model: nn.Module, points: Tensor, labels: Tensor) -> float:
    with torch.no_grad():
        predictions = model(points).argmax(dim=1)
    return float((predictions == labels).float().mean())


def fine_tune_teacher(encoder: Encoder, data: ToyData, epochs: int = 120) -> tuple[Classifier, float]:
    teacher = Classifier(encoder)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.02)
    labeled_points = data.points[data.labeled_indices]
    labeled_labels = data.labels[data.labeled_indices]
    for _ in range(epochs):
        loss = F.cross_entropy(teacher(labeled_points), labeled_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return teacher, accuracy(teacher, data.points, data.labels)


def distill_student(teacher: Classifier, data: ToyData, epochs: int = 160, temperature: float = 2.0) -> tuple[Classifier, float]:
    student = Classifier(Encoder(hidden=16, representation_dim=16))
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
    unlabeled_points = data.points[data.unlabeled_indices]
    with torch.no_grad():
        teacher_targets = F.softmax(teacher(unlabeled_points) / temperature, dim=1)
    for _ in range(epochs):
        student_log_probs = F.log_softmax(student(unlabeled_points) / temperature, dim=1)
        loss = F.kl_div(student_log_probs, teacher_targets, reduction="batchmean") * temperature**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return student, accuracy(student, data.points, data.labels)


def run(seed: int = 7) -> dict[str, float]:
    torch.manual_seed(seed)
    data = make_data(seed)
    encoder, contrastive_loss = pretrain_encoder(data, seed)
    teacher, teacher_accuracy = fine_tune_teacher(encoder, data)
    _, student_accuracy = distill_student(teacher, data)
    return {
        "contrastive_loss": contrastive_loss,
        "teacher_accuracy": teacher_accuracy,
        "student_accuracy": student_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    metrics = run(args.seed)
    print("=== SimCLRv2 toy three-stage workflow ===")
    print("[stage 1] contrastive pretraining on unlabeled views")
    print(f"  final NT-Xent loss: {metrics['contrastive_loss']:.4f}")
    print("[stage 2] fine-tuning a teacher with four labels per class")
    print(f"  teacher accuracy on all synthetic points: {metrics['teacher_accuracy']:.3f}")
    print("[stage 3] distilling teacher predictions on unlabeled points")
    print(f"  student accuracy on all synthetic points: {metrics['student_accuracy']:.3f}")


if __name__ == "__main__":
    main()
