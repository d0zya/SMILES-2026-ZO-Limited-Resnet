from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

import head_init
from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True
TRAIN_SAMPLES_PER_CLASS = 80

def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,

):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    head_init.configure_imprinting(data_dir)
    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY, # True
        download=True,
        transform=get_transforms(train=True),
    )
    indices_by_class = [[] for _ in range(100)]
    for idx, target in enumerate(full_train_dataset.targets):
        if len(indices_by_class[target]) < TRAIN_SAMPLES_PER_CLASS:
            indices_by_class[target].append(idx)
        if all(len(indices) >= TRAIN_SAMPLES_PER_CLASS for indices in indices_by_class):
            break

    selected_indices = [idx for indices in indices_by_class for idx in indices]
    train_dataset = Subset(full_train_dataset, selected_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_dataset, train_loader
