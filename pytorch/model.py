import torch
from torch import nn

import numpy as np

from dataclasses import dataclass

kwds = {
    "kernel_size": 4,
    "stride": 2,
    "padding": 1,
    "bias": False
}
N_CH = 16


def get_discriminator():
    return nn.Sequential(
        nn.Conv2d(1, N_CH, **kwds),  # (14,14)
        nn.BatchNorm2d(N_CH),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH, N_CH * 2, **kwds),  # (7,7)
        nn.BatchNorm2d(N_CH * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH * 2, N_CH * 4, kernel_size=3,
                  stride=1, padding=1, bias=False),  # (7,7)
        nn.BatchNorm2d(N_CH * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH * 4, N_CH * 8, kernel_size=3,
                  stride=1, padding=0, bias=False),  # (5,5)
        nn.BatchNorm2d(N_CH * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH * 8, 2, kernel_size=1, stride=1, padding=0),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )


def get_generator():
    return nn.Sequential(
        nn.Conv2d(1, N_CH, **kwds),  # (14,14)
        nn.BatchNorm2d(N_CH),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH, N_CH * 2, **kwds),  # (7,7)
        nn.BatchNorm2d(N_CH * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH * 2, N_CH * 4, kernel_size=3,
                  stride=1, padding=1, bias=False),  # (7,7)
        nn.BatchNorm2d(N_CH * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(N_CH * 4, N_CH * 8, kernel_size=3,
                  stride=1, padding=0, bias=False),  # (5,5)
        nn.BatchNorm2d(N_CH * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(N_CH * 8, N_CH * 4, kernel_size=3,
                           stride=1, padding=0, bias=False),    # (7,7)
        nn.BatchNorm2d(N_CH * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(N_CH * 4, N_CH * 2, kernel_size=3,
                           stride=1, padding=1, bias=False),    # (7,7)
        nn.BatchNorm2d(N_CH * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(N_CH * 2, N_CH, **kwds),  # (14,14)
        nn.BatchNorm2d(N_CH),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(N_CH, 1, kernel_size=4, stride=2, padding=1),    # (28,28)
        nn.Sigmoid()
    )


@dataclass
class Detector(nn.Module):
    """
    テスト用
    GeneratorとDiscriminatorを保持して分類器のように振る舞う
    """
    gen: nn.Module
    dis: nn.Module
    noise_std: float
    device: torch.device

    def __post_init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        noise = torch.from_numpy(noise).float().to(self.device)
        x_noise = (x + noise).clamp(0.0, 1.0)
        x_recon = self.gen(x_noise)
        y = self.dis(x_recon)

        return y
