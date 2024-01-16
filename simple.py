import torch
import torchvision
import wandb

learning_rate = 1e-3
batch_size = 100
epochs = 5

wandb.init(
    project="demo",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": epochs,
    "batch_size": batch_size,
    }
)

training_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
        )

test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
        )

train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

train_len = len(train_dataloader.dataset)
test_len = len(test_dataloader.dataset)
test_batches = len(test_dataloader)

class NeuralNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
                torch.nn.Conv2d(3, 6, kernel_size=8, stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(6, 12, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(300, 10)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

model = NeuralNetwork()
print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(epochs):
        print("Epoch {}\n-------------------------------".format(e+1))
        model.train()
        for batch, (x,y) in enumerate(train_dataloader):
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if batch % 100 == 0:
                        loss, current = loss.item(), batch * len(x)
                        print("Loss: {} [{}%]".format( loss, str(100*current/train_len)[:4]))
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
                for x,y in test_dataloader:
                        pred = model(x)
                        test_loss += loss_fn(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= test_batches
        correct /= test_len
        print("[Test] Accuracy: {}%, Loss: {} \n".format(str(100*correct)[:4], test_loss))
        
        wandb.log({"acc": 100*correct, "loss": test_loss})
                                
wandb.finish()
