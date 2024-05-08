import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.datasets import make_swiss_roll

import torchvision

def savefig(fname: str, show_figure: bool = True) -> None:
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(
    train_losses: np.ndarray, test_losses: np.ndarray, title: str, fname: str
) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label="train loss")
    plt.plot(x_test, test_losses, label="test loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    savefig(fname)


def q1_data(n=100000):
    x, _ = make_swiss_roll(n, noise=0.5)
    x = x[:, [0, 2]]
    return x.astype('float32')


def visualize_q1_dataset():
    data = q1_data()
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def save_multi_scatter_2d(data: np.ndarray) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            axs[i, j].scatter(data[i * 3 + j, :, 0], data[i * 3 + j, :, 1])
    plt.title("Q1 Samples")
    savefig("results/q1_samples.png")


def show_samples(
    samples: np.ndarray, fname: str = None, nrow: int = 10, title: str = "Samples"
):
    import torch
    from torchvision.utils import make_grid

    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def q1_save_results(fn):
    train_data = q1_data(n=100000)
    test_data = q1_data(n=10000)
    train_losses, test_losses, samples = fn(train_data, test_data)

    print(f"Final Test Loss: {test_losses[-1]:.4f}")

    save_training_plot(
        train_losses,
        test_losses,
        "Q1 Train Plot",
        "results/q1_train_plot.png"
    )

    save_multi_scatter_2d(samples)


def load_q2_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    test_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=False)
    return train_data, test_data


def visualize_q2_data():
    train_data, _ = load_q2_data()
    imgs = train_data.data[:100]
    show_samples(imgs, title='CIFAR-10 Samples')


def q2_save_results(fn):
    train_data, test_data = load_q2_data()
    train_data = train_data.data / 255.0
    test_data = test_data.data / 255.0
    train_losses, test_losses, samples = fn(train_data, test_data)
    
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        "Q2 Train Plot",
        "results/q2_train_plot.png"
    ) 

    samples = samples.reshape(-1, *samples.shape[2:])
    show_samples(samples * 255.0, fname="results/q2_samples.png", title="Q2 CIFAR-10 generated samples")
