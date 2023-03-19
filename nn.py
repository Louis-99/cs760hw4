import torch 
import torchvision
from torchvision.datasets import MNIST
from torch.utils import data



LR = 0.01

# Change this 
RANDOM_INIT = True

class Linear:
    def __init__(self, d1, d2):
        if RANDOM_INIT:
            self.w = torch.rand((d1, d2)) * 2 - 1
            self.b = torch.rand(d1) * 2 - 1
        else:
            self.w = torch.zeros((d1, d2))
            self.b = torch.zeros(d1)
    
    def forward(self, x):
        self.x = x
        # print(f"{self.w.shape=}, {x.shape=}")
        self.y = torch.einsum('ab,cb->ca',[self.w, x]) + self.b
        return self.y
    
    def backward(self, dy):
        if torch.sum(torch.isnan(dy)) > 0: 
            raise Exception("nan at 21")
        # print(f"{dy.shape=}, {self.x.shape=}")
        dw = torch.einsum('ca,cb->cab', [dy, self.x])
        # print(f"{dw.shape=}")
        dx = torch.matmul(dy, self.w)
        # print(f"{dx.shape=}")
        self.w -= LR * torch.mean(dw, dim=0)
        self.b -= LR * torch.mean(dy, dim=0)
        if torch.sum(torch.isnan(dx)) > 0: 
            raise Exception("nan at 30")
        return dx
    
class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.y = torch.sigmoid(x)
        return self.y
    
    def backward(self, dy):
        if torch.sum(torch.isnan(dy)) > 0: 
            raise Exception("nan at 20")

        d_sigmoid = self.y * (1 - self.y)
        if torch.sum(torch.isnan(d_sigmoid)) > 0: 
            raise Exception("nan at 40")
        return dy * d_sigmoid
    
class Softmax:
    def __init__(self):
        pass
    
    def forward(self, x):
        self.x = x
        self.y = torch.softmax(x, 1)
        return self.y
    
    def backward(self, dy):
        if torch.sum(torch.isnan(dy)) > 0: 
            raise Exception("nan at 57")

        d1 = torch.einsum('ca,cb->cab', [self.y, self.y])
        d2 = torch.diag_embed(self.y, dim1=-2, dim2=-1)
        d_softmax = d2-d1
        # print(f"{self.y.shape=}, {dy.shape=}, {d_softmax.shape=}")
        ret = torch.einsum('ab,abc->ac', dy, d_softmax)
        if torch.sum(torch.isnan(ret)) > 0: 
            raise Exception("nan at 68")
        return ret
    

def cross_entropy(y, yhat):
    loss = - torch.mean(torch.einsum('ab,ab->a', [y, torch.log(yhat + 1e-20)]))
    dy = -y/ (yhat + 1e-20)
    if torch.sum(torch.isnan(dy)) > 0: 
        raise Exception("nan at 76")
    return loss, dy  

class NN:
    def __init__(self, d, d1, d2, k):
        self.layers = [
            Linear(d1, d),
            Sigmoid(),
            Linear(d2, d1),
            Sigmoid(),
            Linear(k, d2),
            Softmax()
        ]

    def forward(self, x):
        x = torch.flatten(x, start_dim=-3, end_dim=-1)
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)

BATCH_SIZE = 64

def train():
    lss = []
    acc = []
    mnist = MNIST(".", train=True, download=True, transform=torchvision.transforms.ToTensor())
    ds = data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)
    net = NN(28*28, 300, 200, 10)
    for _ in range(10):
        for i, (x, y) in enumerate(ds):
            yhat = net.forward(x)
            label = torch.nn.functional.one_hot(y, 10).type(torch.FloatTensor)
            loss, dy = cross_entropy(label, yhat)
            if i % 20 == 0:
                loss = float(loss)
                lss.append(loss)
                pred = torch.argmax(yhat, 1)
                accuracy = float(torch.sum(y == pred)) / BATCH_SIZE
                acc.append(accuracy)
                print(f"{loss=}, {accuracy=}")
                # print(net.layers[1].y)
            net.backward(dy)

    print(f"{lss=}")
    print(f"{acc=}")
    return net

def test(net):
    mnist = MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
    ds = data.DataLoader(mnist, batch_size=BATCH_SIZE)
    total_batch = len(ds)
    correct_num = 0
    for (x, y) in ds:
        yhat = net.forward(x)
        pred = torch.argmax(yhat, 1)
        correct_num += int(torch.sum(y == pred))
    accuracy = correct_num / (total_batch * BATCH_SIZE)
    print(f"test: {accuracy=}")


if __name__ == '__main__':
    net = train()
    test(net)