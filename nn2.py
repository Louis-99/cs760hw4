import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils import data

LR = 1
BATCH_SIZE = 64

class NN2(nn.Module):
    def __init__(self,  d, d1, d2, k):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(d, d1),
            nn.Sigmoid(),
            nn.Linear(d1, d2),
            nn.Sigmoid(),
            nn.Linear(d2, k),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
    


def train(ds, net):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr=LR)

    lss = []
    
    net.train()
    for _ in range(10):
        for i, (x, y) in enumerate(ds):
            pred = net(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if i % 20 == 0:
                loss = loss.item()
                lss.append(loss)
                print(f"{loss=}")

    
    print(f"{lss=}")

def test(ds, net):
    total_batch = len(ds)
    net.eval()
    correct_num = 0
    with torch.no_grad():
        for x, y in ds:
            yhat = net(x)
            pred = torch.argmax(yhat, 1)
            correct_num += int(torch.sum(y == pred))

    accuracy = correct_num / (total_batch * BATCH_SIZE)
    print(f"test: {accuracy=}")





def main():
    net = NN2(28*28, 300, 200, 10)

    mnist_train = MNIST(".", train=True, download=True, transform=torchvision.transforms.ToTensor())
    ds_train = data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    mnist_test = MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
    ds_test = data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    train(ds_train, net)
    test(ds_test, net)


if __name__ == '__main__':
    main()