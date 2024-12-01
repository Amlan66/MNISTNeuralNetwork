import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse

def get_dataloaders(ci_mode=False):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    batch_size = 32

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Use full MNIST dataset (60000 training, 10000 test)
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    test_dataset = datasets.MNIST('../data', train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    
    # No dataset reduction even in CI mode
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc= f'epoch={epoch} loss={loss.item()} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy

def train_model(model, device, optimizer, epochs=20, ci_mode=False):
    train_loader, test_loader = get_dataloaders(ci_mode)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10.0,
        three_phase=False,
        final_div_factor=100,
    )

    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        accuracy = test(model, device, test_loader)
        best_accuracy = max(best_accuracy, accuracy)
    
    return best_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--ci-mode', action='store_true',
                        help='run in CI mode with reduced dataset')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Import your model from model.py
    from model import Net
    
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    accuracy = train_model(model, device, optimizer, 
                          epochs=args.epochs, ci_mode=args.ci_mode)
    print(f"Best accuracy: {accuracy:.2f}%") 