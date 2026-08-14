"""Educational SimCLR walkthrough for one batch of five images.

This is a small toy prototype, not an ImageNet-scale reproduction. It demonstrates
one complete forward pass:

    5 images -> 10 augmented views -> ResNet encoder -> projection MLP
              -> pairwise similarities -> NT-Xent cross-entropy loss

Run from the paper directory with:

    uv run --locked python prototypes/five_image_batch/run_simclr_batch.py
"""

from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import models, transforms


# Stable public image URLs from Lorem Picsum. We randomly sample five from this
# pool, but keep the URLs fixed so the experiment is reproducible with a seed.
IMAGE_URL_POOL = [
    "https://picsum.photos/id/237/512/512",
    "https://picsum.photos/id/1025/512/512",
    "https://picsum.photos/id/1069/512/512",
    "https://picsum.photos/id/1074/512/512",
    "https://picsum.photos/id/1084/512/512",
    "https://picsum.photos/id/1080/512/512",
    "https://picsum.photos/id/219/512/512",
    "https://picsum.photos/id/292/512/512",
    "https://picsum.photos/id/433/512/512",
    "https://picsum.photos/id/488/512/512",
]


class SimCLRTransform:
    """Create one stochastic SimCLR-style view from an input PIL image."""

    def __init__(self, image_size: int) -> None:
        # The paper emphasizes compositions of augmentations. Each call to this
        # transform samples fresh random choices, so the same image gets two
        # different views below.
        color_jitter = transforms.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.6, 1.0),
                    ratio=(0.75, 1.3333),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> Tensor:
        return self.transform(image)


def choose_image_urls(seed: int, count: int) -> list[str]:
    """Choose stable-but-random image URLs for this run."""
    if count > len(IMAGE_URL_POOL):
        raise ValueError(f"count must be <= {len(IMAGE_URL_POOL)}")
    rng = random.Random(seed)
    return rng.sample(IMAGE_URL_POOL, k=count)


def download_images(urls: list[str], data_dir: Path) -> list[Path]:
    """Download public images once and reuse them on later runs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for index, url in enumerate(urls):
        destination = data_dir / f"image_{index}.jpg"
        if not destination.exists():
            print(f"[download] {url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "simclr-paper-prototype/0.1"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                destination.write_bytes(response.read())
        else:
            print(f"[cache]    {destination}")
        image_paths.append(destination)

    return image_paths


def load_images(image_paths: list[Path]) -> list[Image.Image]:
    """Load images as RGB PIL images and print their original dimensions."""
    images: list[Image.Image] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        print(f"[image]    {path.name}: original size={image.size}, mode={image.mode}")
        images.append(image)
    return images


def save_augmented_views(
    image_paths: list[Path],
    view_1: Tensor,
    view_2: Tensor,
    views_dir: Path,
) -> None:
    """Save the two normalized tensors as view_1.png/view_2.png per image."""
    # The model receives normalized tensors, so undo ImageNet normalization
    # before converting them back to viewable PNG files.
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    to_pil = transforms.ToPILImage()

    for image_path, first_view, second_view in zip(
        image_paths, view_1, view_2, strict=True
    ):
        image_views_dir = views_dir / image_path.stem
        image_views_dir.mkdir(parents=True, exist_ok=True)
        for view_number, view in enumerate((first_view, second_view), start=1):
            view_for_saving = (view.cpu() * std + mean).clamp(0.0, 1.0)
            output_path = image_views_dir / f"view_{view_number}.png"
            to_pil(view_for_saving).save(output_path)
            print(f"  saved: {output_path.relative_to(views_dir.parent)}")



def build_encoder_and_projector() -> tuple[nn.Module, nn.Module, int, int]:
    """Build a randomly initialized ResNet encoder and a two-layer MLP projector."""
    # weights=None is intentional: this prototype demonstrates SimCLR
    # pretraining from scratch rather than using supervised ImageNet weights.
    resnet = models.resnet18(weights=None)
    encoder_dim = resnet.fc.in_features
    resnet.fc = nn.Identity()

    projection_dim = 128
    projector = nn.Sequential(
        nn.Linear(encoder_dim, encoder_dim),
        nn.ReLU(inplace=True),
        nn.Linear(encoder_dim, projection_dim),
    )
    return resnet, projector, encoder_dim, projection_dim


def nt_xent_loss(z: Tensor, temperature: float) -> tuple[Tensor, Tensor, Tensor]:
    """Compute SimCLR's bidirectional NT-Xent loss.

    Views are arranged as [view_1_for_all_images, view_2_for_all_images].
    Thus, the positive for anchor i is (i + N) modulo 2N.
    """
    batch_size_2 = z.shape[0]
    if batch_size_2 % 2 != 0:
        raise ValueError("Expected an even number of views")
    n = batch_size_2 // 2

    # Normalize embeddings so a dot product becomes cosine similarity.
    z_normalized = F.normalize(z, dim=1)
    similarity = z_normalized @ z_normalized.T

    # An anchor must not select itself as its positive candidate.
    self_mask = torch.eye(batch_size_2, dtype=torch.bool, device=z.device)
    logits = similarity.masked_fill(self_mask, float("-inf")) / temperature

    # For anchor i, the paired view is i+N for the first half and i-N for
    # the second half. This is the automatically generated target label.
    labels = (torch.arange(batch_size_2, device=z.device) + n) % batch_size_2
    loss = F.cross_entropy(logits, labels)
    return loss, logits, labels


def print_anchor_explanation(logits: Tensor, labels: Tensor, anchor: int) -> None:
    """Print one row as the multiclass classification problem used by NT-Xent."""
    probabilities = logits[anchor].softmax(dim=0)
    positive = int(labels[anchor].item())
    top_values, top_indices = probabilities.topk(k=min(5, probabilities.numel()))

    print("\n[anchor walkthrough]")
    print(f"  anchor index:             {anchor}")
    print(f"  positive target index:    {positive}")
    print(f"  positive probability:     {probabilities[positive].item():.6f}")
    print("  highest candidate probabilities:")
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        marker = "  <-- positive" if index == positive else ""
        print(f"    candidate {index:2d}: {value:.6f}{marker}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("=== SimCLR: one batch of five images ===")
    print(f"[setup]    seed={args.seed}, image_size={args.image_size}")
    print(f"[setup]    temperature={args.temperature}")
    print(f"[setup]    device={device}")

    prototype_dir = Path(__file__).resolve().parent
    image_urls = choose_image_urls(seed=args.seed, count=5)
    image_paths = download_images(image_urls, prototype_dir / "data")
    images = load_images(image_paths)

    print("\n[step 1] Creating two random augmented views per image")
    view_transform = SimCLRTransform(image_size=args.image_size)
    view_1 = torch.stack([view_transform(image) for image in images])
    view_2 = torch.stack([view_transform(image) for image in images])
    print(f"  view_1 tensor shape: {tuple(view_1.shape)}")
    print(f"  view_2 tensor shape: {tuple(view_2.shape)}")
    print("\n[step 1b] Saving generated views grouped by source image")
    save_augmented_views(
        image_paths=image_paths,
        view_1=view_1,
        view_2=view_2,
        views_dir=prototype_dir / "views",
    )

    # Concatenate the two views. The first five and second five positions are
    # paired by index: 0<->5, 1<->6, ..., 4<->9.
    all_views = torch.cat([view_1, view_2], dim=0).to(device)
    print(f"  all_views tensor shape: {tuple(all_views.shape)}")

    print("\n[step 2] ResNet encoder forward pass")
    encoder, projector, encoder_dim, projection_dim = build_encoder_and_projector()
    encoder = encoder.to(device)
    projector = projector.to(device)
    print(f"  encoder: ResNet-18 with final classifier removed")
    print(f"  encoder output h dimension: {encoder_dim}")

    with torch.no_grad():
        h = encoder(all_views)
        z = projector(h)
    print(f"  h shape: {tuple(h.shape)}")
    print(f"  z shape: {tuple(z.shape)}")
    print("  The contrastive loss is applied to z; h is the representation kept downstream.")

    print("\n[step 3] Pairwise similarities and NT-Xent loss")
    loss, logits, labels = nt_xent_loss(z, temperature=args.temperature)
    print(f"  similarity/logit matrix shape: {tuple(logits.shape)}")
    print(f"  labels shape: {tuple(labels.shape)}")
    print(f"  labels: {labels.tolist()}")
    print("  label 5 for anchor 0 means view_2 of image 0 is its positive.")
    print("  every other non-self view is a candidate negative for that anchor.")
    print_anchor_explanation(logits, labels, anchor=0)
    print(f"\n[result] NT-Xent loss: {loss.item():.6f}")
    print("[result] Since the encoder is randomly initialized, this loss is only a forward-pass demonstration.")
    print("[result] Training would backpropagate this loss and update both the projector and encoder.")


if __name__ == "__main__":
    main()
