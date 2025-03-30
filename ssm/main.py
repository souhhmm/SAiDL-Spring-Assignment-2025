import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from s4b import S4Model
from tqdm import tqdm
import wandb
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key="2f3ffd7baf545af396e18e48bfa20b33d2609dcc")

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Lambda(lambda x: x.reshape(3, -1).t())
])

trainset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=True, download=True, transform=transform)
trainset, _ = split_train_val(trainset, val_split=0.1)

valset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=True, download=True, transform=transform)
_, valset = split_train_val(valset, val_split=0.1)

testset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=False, download=True, transform=transform)

d_input = 3
d_output = 10
batch_size = 64

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

model = S4Model(
    d_input=d_input,
    d_model=512,
    d_output=d_output,
    n_blocks=6,
    n=64,
    l_max=1024,
    dropout=0.2,
)

model.to(device)

num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


wandb.init(
    project="saidl-s4",
    name = f"s4_run_{datetime.now().strftime('%d%m_%H%M')}",
    config={
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "epochs": num_epochs,
        "batch_size": 64,
        "model_config": {
            "d_model": 128,
            "n_blocks": 6,
            "dropout": 0.2
        }
    }
)

def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, desc='training')
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss = (running_loss * i + loss.item()) / (i + 1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        current_acc = correct / total

        wandb.log({
            "train/batch_loss": loss.item(),
            "train/running_loss": running_loss,
            "train/running_acc": current_acc
        })

        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'acc': f'{correct/total:.2f}'
        })

    return running_loss, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='evaluating')
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss = (running_loss * i + loss.item()) / (i + 1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss:.4f}',
                'acc': f'{correct/total:.2f}'
            })

    return running_loss, correct / total

best_val_acc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, valloader, criterion, device)
    scheduler.step()

    wandb.log({
        "train/epoch_loss": train_loss,
        "train/epoch_acc": train_acc,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "epoch": epoch
    })

    print(f'epoch: {epoch+1}/{num_epochs}')
    print(f'train loss: {train_loss:.4f} | train acc: {train_acc:.2f}')
    print(f'val loss: {val_loss:.4f} | val acc: {val_acc:.2f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    print('-' * 50)

model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, testloader, criterion, device)
wandb.log({
    "test/loss": test_loss,
    "test/acc": test_acc
})
print(f'test loss: {test_loss:.4f} | test acc: {test_acc:.2f}')

wandb.finish()
