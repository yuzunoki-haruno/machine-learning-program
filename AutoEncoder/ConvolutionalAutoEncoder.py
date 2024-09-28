import torch
from torch import Tensor, nn


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_channel, out_channel, kernel_size = 1, 16, 3
        self.encoder = Encoder(in_channel, out_channel, kernel_size)
        self.decoder = Decoder(out_channel, in_channel, kernel_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=1)
        self.layer2 = nn.MaxPool2d(kernel_size - 1)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.selu(self.layer1(x))
        return self.layer2(h)


class Decoder(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int) -> None:
        super().__init__()
        self.layer1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=2, padding=1, output_padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.layer1(x))


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST

    DATA_PATH = "/mnt/pg-nas/dataset/"

    # get hyper parameters.
    parser = argparse.ArgumentParser(prog="ConvolutionalAutoEncoder.py")
    parser.add_argument("epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cpu" if args.cpu else "cuda"

    # setup train data.
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(DATA_PATH, download=True, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_shape = (28, 28)

    # setup autoencoder.
    model = ConvolutionalAutoEncoder()
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    model.to(device)

    # train model
    model.train()
    for epoch in range(epochs):
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
