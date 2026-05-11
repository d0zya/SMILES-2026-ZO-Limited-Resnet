"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonl, small-scale random, learned bias init).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_IMPRINT_SAMPLES_PER_CLASS = 100
_TRIM_KEEP_RATIO = 0.8

_imprinting_data_dir: str | None = None
_imprinting_weights_cache: torch.Tensor | None = None


def configure_imprinting(data_dir: str) -> None:
    """Store the dataset root so head init can build data-driven prototypes."""
    global _imprinting_data_dir
    _imprinting_data_dir = data_dir


def _fallback_init(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    with torch.no_grad():
        layer.weight.div_(layer.weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
    nn.init.zeros_(layer.bias)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_imprinting_weights(num_classes: int, in_features: int) -> torch.Tensor | None:
    global _imprinting_weights_cache

    if _imprinting_weights_cache is not None:
        if tuple(_imprinting_weights_cache.shape) == (num_classes, in_features):
            return _imprinting_weights_cache
        _imprinting_weights_cache = None

    if _imprinting_data_dir is None:
        return None

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )
    dataset = datasets.CIFAR100(
        root=_imprinting_data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    indices_by_class = [[] for _ in range(num_classes)]
    for idx, target in enumerate(dataset.targets):
        if len(indices_by_class[target]) < _IMPRINT_SAMPLES_PER_CLASS:
            indices_by_class[target].append(idx)
        if all(len(indices) >= _IMPRINT_SAMPLES_PER_CLASS for indices in indices_by_class):
            break

    selected_indices = [idx for indices in indices_by_class for idx in indices]
    if not selected_indices:
        return None

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = _get_device()
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)

    features_by_class: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in loader:
            features = backbone(images.to(device)).cpu()
            for feature, label in zip(features, labels):
                features_by_class[int(label)].append(feature)

    prototypes = []
    for class_features in features_by_class:
        if not class_features:
            return None

        class_tensor = torch.stack(class_features, dim=0)
        center0 = class_tensor.mean(dim=0, keepdim=True)
        distances = torch.norm(class_tensor - center0, dim=1)

        keep_count = max(1, int(class_tensor.size(0) * _TRIM_KEEP_RATIO))
        keep_idx = torch.argsort(distances)[:keep_count]
        trimmed_tensor = class_tensor[keep_idx]

        prototype = trimmed_tensor.mean(dim=0)
        prototypes.append(prototype)

    prototypes = torch.stack(prototypes, dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    _imprinting_weights_cache = prototypes
    return _imprinting_weights_cache


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------
    imprint_weights = _get_imprinting_weights(
        num_classes=layer.out_features,
        in_features=layer.in_features,
    )
    if imprint_weights is None:
        _fallback_init(layer)
        return

    with torch.no_grad():
        layer.weight.copy_(imprint_weights)
        nn.init.zeros_(layer.bias)
    # -------------------------------------------------------------------------
