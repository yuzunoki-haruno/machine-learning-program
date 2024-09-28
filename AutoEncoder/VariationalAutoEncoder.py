from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, latent_size: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_size, embedding_size)
        self.decoder = Decoder(embedding_size, input_size)
        self.latent = nn.Linear(latent_size, embedding_size)
        self.avg = nn.Linear(embedding_size, latent_size)
        self.var = nn.Linear(embedding_size, latent_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(x)
        avg = self.avg(h)
        var = F.softplus(self.var(h))
        z = self.latent_varialbe(avg, var)
        z = torch.relu(self.latent(z))
        y = self.decoder(z)
        return y, avg, var

    def latent_variable(self, avg: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn_like(avg)
        return avg + torch.sqrt(var) * eps


def lower_bound(x: Tensor, y: Tensor, avg: Tensor, var: Tensor) -> Tensor:
    bce = F.binary_cross_entropy(y, x, reduction="sum")
    kld = -torch.sum(1 + torch.log(var) - avg**2 - var) / 2
    return bce + kld


class Encoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(input_size, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.layer(x))


class Decoder(torch.nn.Module):
    def __init__(self, embedding_size: int, input_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(embedding_size, input_size)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.layer(x))


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST

    DATA_PATH = "/mnt/pg-nas/dataset/"

    # get hyper parameters.
    parser = argparse.ArgumentParser(prog="VariationalAutoEncoder.py")
    parser.add_argument("embedding_size", type=int)
    parser.add_argument("latent_size", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    embedding_size = args.embedding_size
    latent_size = args.latent_size
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cpu" if args.cpu else "cuda"

    # setup train data.
    transform = transforms.Compose([transforms.ToTensor(), torch.nn.Flatten()])
    dataset = MNIST(DATA_PATH, download=True, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_shape = (28, 28)
    input_size = input_shape[0] * input_shape[1]

    # setup autoencoder.
    model = VariationalAutoEncoder(input_size, embedding_size, latent_size)
    optimizer = optim.Adam(model.parameters())
    model.to(device)

    # train model
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            pred, avg, var = model(x)
            loss = lower_bound(x, pred, avg, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader)
        print(f"{epoch+1:03}/{epochs:03} epochs: Loss {train_loss:.6e}")

    # setup test data.
    dataset = MNIST(DATA_PATH, download=True, train=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    loss = 0
    fig = plt.figure()
    n_rows, n_cols = 2, 4
    model.eval()
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        pred, avg, var = model(x)
        loss += lower_bound(x, pred, avg, var)
        if i < n_cols:
            ax = fig.add_subplot(n_rows, n_cols, 2 * i + 1)
            ax.imshow(x.view(input_shape).cpu().detach().numpy(), cmap="gray")
            ax.axis("off")
            ax.set_title("Original")
            ax = fig.add_subplot(n_rows, n_cols, 2 * i + 2)
            ax.imshow(pred.view(input_shape).cpu().detach().numpy(), cmap="gray")
            ax.axis("off")
            ax.set_title("Decoded")
    fig.tight_layout()
    plt.savefig("TeainingResult.png")

    loss /= len(dataloader)
    print(f"Loss {loss:.6e}")
