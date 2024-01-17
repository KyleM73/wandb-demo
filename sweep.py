import itertools
import torch
import torchvision
import wandb

max_layers = 4
layer_sizes = [32,64,128,256,512,1024]
    
sweep_config = {
    "method" : "bayes", #options: "random", "bayes", "grid" (bayes requires an additional config, "metric" to be set; see below)
    "name" : "sweep",
    "metric" : {"goal" : "minimize", "name" : "test_loss"}, #options: "minimize", "maximize", "target" (target requires an additional param, "target", to be set)
    "run_cap" : 20, #max number of sweeps to run
    "parameters" : {
        "batch_size": {"distribution": "q_uniform", "min" : 50, "max" : 1000, "q" : 10},
        "epochs": {"distribution": "q_uniform", "min" : 5, "max" : 50, "q" : 5},
        "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0.0001},
        "layers" : {"values" : [list(l) for nl in range(1,max_layers+1) for l in itertools.product(layer_sizes, repeat=nl)]},
    }
}

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

def main():
    wandb.init(
        project="demo",
        config={
        "architecture": "MLP",
        "dataset": "MNIST",
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

    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=wandb.config.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=wandb.config.batch_size)

    train_len = len(train_dataloader.dataset)
    test_len = len(test_dataloader.dataset)
    
    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    
    model = NeuralNetwork(layers=wandb.config.layers)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    for e in range(wandb.config.epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for x,y in train_dataloader:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss /= train_batches
        train_correct /= train_len
        
        model.eval()
        test_loss, test_correct = 0, 0
        with torch.no_grad():
                for x,y in test_dataloader:
                        pred = model(x)
                        test_loss += loss_fn(pred, y).item()
                        test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= test_batches
        test_correct /= test_len
        
        wandb.log(
            {
                "epoch": e,
                "train_acc": train_correct,
                "train_loss": train_loss,
                "test_acc": test_correct,
                "test_loss": test_loss,
                }
            )

sweep_id = wandb.sweep(sweep=sweep_config, project="demo")
wandb.agent(sweep_id, function=main, count=sweep_config["run_cap"])
