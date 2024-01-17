import torch
import torchvision
import wandb

learning_rate = 1e-3
batch_size = 100
epochs = 5

wandb.init(
    project="demo",
    config={
    "learning_rate": learning_rate,
    "architecture": "MLP",
    "dataset": "MNIST",
    "epochs": epochs,
    "batch_size": batch_size,
    }
)

training_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
        )

test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
        )

train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

train_len = len(train_dataloader.dataset)
test_len = len(test_dataloader.dataset)

train_batches = len(train_dataloader)
test_batches = len(test_dataloader)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, inp : int = 784, layers: list = [128,128,128]) -> None:
        super().__init__()
        layer_list = [torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(inp, layers[0]), torch.nn.ReLU())]
        for i in range(len(layers)-1):
                layer_list.append(torch.nn.Sequential(torch.nn.Linear(layers[i],layers[i+1]), torch.nn.ReLU()))
        layer_list.append(torch.nn.Sequential(torch.nn.Linear(layers[-1], 10), torch.nn.Softmax(dim=1)))
        self.network = torch.nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

model = NeuralNetwork()
print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(epochs):
        print("Epoch {}\n-------------------------------".format(e+1))
        model.train()
        train_loss, train_correct = 0, 0
        for batch, (x,y) in enumerate(train_dataloader):
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                if batch % 100 == 0:
                        loss, current = loss.item(), batch * len(x)
                        print("Loss: {} [{}%]".format( loss, str(100*current/train_len)[:4]))
        train_loss /= train_batches
        train_correct /= train_len
        print("[Train] Accuracy: {}%, Loss: {} \n".format(str(100*train_correct)[:4], train_loss))
        model.eval()
        test_loss, test_correct = 0, 0
        with torch.no_grad():
                for x,y in test_dataloader:
                        pred = model(x)
                        test_loss += loss_fn(pred, y).item()
                        test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= test_batches
        test_correct /= test_len
        print("[Test] Accuracy: {}%, Loss: {} \n".format(str(100*test_correct)[:4], test_loss))
        
        wandb.log(
                {
                        "train_acc" : train_correct,
                        "train_loss" : train_loss,
                        "test_acc": test_correct,
                        "test_loss": test_loss,
                        }
                )
                                
wandb.finish()
