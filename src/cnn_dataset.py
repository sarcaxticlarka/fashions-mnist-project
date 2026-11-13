"""PyTorch Dataset and DataLoader wrappers for Fashion-MNIST (Phase 2).

Maintains exact parity with Phase 1:
- Splits: 48,000 train, 12,000 validation, 10,000 test.
- Normalization: scaled to [0, 1] float32.
- Data augmentation (for ablation study): RandomHorizontalFlip, RandomCrop with 2-px reflection padding.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import SEED
from .data_loader import make_splits, Splits


class FashionMNISTDataset(Dataset):
    """PyTorch Dataset wrapping Fashion-MNIST numpy arrays (N, 28, 28) scaled to [0, 1]."""
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        # Convert (N, 28, 28) -> (N, 1, 28, 28) float32 torch tensor
        self.X = torch.from_numpy(X[:, np.newaxis, :, :]).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        target = self.y[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_augment_transform(random_erasing: float = 0.0):
    """Standard spatial augmentation for 28x28 clothing items.

    random_erasing > 0 additionally blanks a random rectangle with that probability
    (forces the network to use more than one cue, e.g. not only the collar)."""
    t = [transforms.RandomCrop(28, padding=2, padding_mode="reflect"),
         transforms.RandomHorizontalFlip(p=0.5)]
    if random_erasing > 0:
        t.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.15), value=0.0))
    return transforms.Compose(t)


def build_dataloaders(S: Splits, batch_size: int = 128, augment_train: bool = False,
                      num_workers: int = 0, seed: int = SEED, random_erasing: float = 0.0):
    """Build train, validation, trainval, and test DataLoaders."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_tf = get_augment_transform(random_erasing) if augment_train else None

    ds_tr = FashionMNISTDataset(S.X_tr, S.y_tr, transform=train_tf)
    ds_val = FashionMNISTDataset(S.X_val, S.y_val, transform=None)
    ds_trval = FashionMNISTDataset(S.X_trainval, S.y_trainval, transform=train_tf)
    ds_test = FashionMNISTDataset(S.X_test, S.y_test, transform=None)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g,
                       num_workers=num_workers, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    dl_trval = DataLoader(ds_trval, batch_size=batch_size, shuffle=True, generator=g,
                          num_workers=num_workers, pin_memory=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=False)

    return dict(tr=dl_tr, val=dl_val, trval=dl_trval, test=dl_test)

