import torch
from torch import Tensor, nn


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_size, embedding_size)
        self.decoder = Decoder(embedding_size, input_size)

    def forward(self, x):
        y = self.encoder(x)
        return self.decoder(y)


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
    from tqdm import tqdm

    DATA_PATH = "/mnt/pg-nas/dataset/"

    # get hyper parameters.
    parser = argparse.ArgumentParser(prog="SimpleAutoEncoder.py")
    parser.add_argument("embedding_size", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    embedding_size = args.embedding_size
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
    model = SimpleAutoEncoder(input_size, embedding_size)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    model.to(device)

    # train model
    loss_history = list()
    model.train()
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            pred = model(x)
            loss = criterion(pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader)
        loss_history.append(train_loss)
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
        pred = model(x)
        loss += criterion(x, pred)
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
    plt.savefig("TrainingResult.png")

    loss /= len(dataloader)
    print(f"Loss {loss:.6e}")
