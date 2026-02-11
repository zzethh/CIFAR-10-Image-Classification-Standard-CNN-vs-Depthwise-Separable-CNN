import os
import time
import json
import itertools
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

plt.style.use('seaborn-v0_8-darkgrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AdaptiveCNN(nn.Module):
    """Standard CNN or variant with depthwise-separable conv in place of Conv2 and Conv3.
    Assignment required at least two standard conv layers replaced; we chose the second
    and third blocks so the model stays comparable while meeting that requirement."""
    def __init__(self, use_separable=False, dropout_rate=0.5):
        super().__init__()
        self.use_separable = use_separable

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        if use_separable:
            # Replaced Conv2 and Conv3 with depthwise-separable to meet "at least two layers" requirement
            self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3)
            self.conv3 = DepthwiseSeparableConv(64, 128, kernel_size=3)
        else:
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    gen = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size], generator=gen)
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_macs(model, input_size=(1, 3, 32, 32)):
    """MACs = Multiply-Accumulate ops. For Conv2d we use (in_channels // groups) so
    depthwise layers (groups=in_channels) are counted correctly per filter."""
    total_macs = 0
    def hook_fn(module, input, output):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2:]
            # groups: standard conv has groups=1; depthwise has groups=in_channels
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
            total_macs += kernel_ops * module.out_channels * out_h * out_w
        elif isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features
            
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))
            
    dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    return total_macs

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), 100. * correct / total, all_preds, all_targets

def run_grid_search():
    lrs = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    dropouts = [0.2, 0.3, 0.5]
    combos = list(itertools.product(lrs, batch_sizes, dropouts))
    n_combos = len(combos)
    assert n_combos == 27, "Grid must have 3x3x3 = 27 combinations"
    print(f"Grid Search: 3 hyperparameters x 3 values each = {n_combos} combinations (Standard CNN only).")
    best_acc = 0
    best_params = {}
    results = []

    for idx, (lr, bs, drop) in enumerate(combos, start=1):
        train_loader, val_loader, _ = get_data_loaders(bs)
        model = AdaptiveCNN(use_separable=False, dropout_rate=drop).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"  [{idx}/{n_combos}] LR={lr}, batch_size={bs}, dropout={drop}")
        for epoch in range(5):
            train_one_epoch(model, train_loader, criterion, optimizer)
            
        _, val_acc, _, _ = evaluate(model, val_loader, criterion)
        results.append({'lr': lr, 'bs': bs, 'drop': drop, 'acc': val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {'lr': lr, 'batch_size': bs, 'dropout': drop}

    print(f"Grid Search complete: all {n_combos} combinations evaluated.")
    print(f"Grid Search Best: {best_params} with Val Acc: {best_acc:.2f}%")
    return best_params

def optuna_objective(trial):
    lr = trial.suggest_categorical('lr', [0.001, 0.01, 0.0001])
    bs = trial.suggest_categorical('batch_size', [32, 64, 128])
    drop = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    
    train_loader, val_loader, _ = get_data_loaders(bs)
    model = AdaptiveCNN(use_separable=False, dropout_rate=drop).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        train_one_epoch(model, train_loader, criterion, optimizer)
        
    _, val_acc, _, _ = evaluate(model, val_loader, criterion)
    return val_acc

def save_visualizations(history, test_metrics, cm, model_name):
    os.makedirs('visualizations', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='#2ecc71', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', color='#e74c3c', linewidth=2)
    plt.title(f'{model_name} Accuracy', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#9b59b6', linewidth=2)
    plt.title(f'{model_name} Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_learning_curves.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

def final_training(best_params):
    epochs = 15
    bs = best_params['batch_size']
    lr = best_params['lr']
    drop = best_params['dropout']
    
    train_loader, val_loader, test_loader = get_data_loaders(bs)
    criterion = nn.CrossEntropyLoss()
    
    configs = [('Standard_CNN', False), ('Separable_CNN', True)]
    
    metrics_report = {}
    
    for name, use_sep in configs:
        print(f"Training Final Model: {name}")
        model = AdaptiveCNN(use_separable=use_sep, dropout_rate=drop).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
            
            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            print(f"Epoch {epoch+1}/{epochs} - Acc: {t_acc:.2f}% Val Acc: {v_acc:.2f}%")
            
        test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion)
        
        params_count = count_parameters(model)
        macs_count = estimate_macs(model)
        
        prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='weighted')
        cm = confusion_matrix(targets, preds)
        
        metrics_report[name] = {
            'Accuracy': test_acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Parameters': params_count,
            'MACs': macs_count
        }
        
        save_visualizations(history, metrics_report[name], cm, name)
        
    return metrics_report

if __name__ == "__main__":
    set_seed()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    gs_best = run_grid_search()
    
    print("Starting Optuna Search...")
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=10)
    opt_best = study.best_params
    print(f"Optuna Best: {opt_best}")
    
    final_metrics = final_training(opt_best)
    
    print("\nFinal Comparison:")
    for model, metrics in final_metrics.items():
        print(f"--- {model} ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    plt.figure(figsize=(10, 6))
    models = list(final_metrics.keys())
    params = [final_metrics[m]['Parameters'] for m in models]
    macs = [final_metrics[m]['MACs'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    rects1 = ax1.bar(x - width/2, params, width, label='Parameters', color='#e67e22')
    ax1.set_ylabel('Parameter Count', color='#e67e22', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#e67e22')
    ax1.set_title('Efficiency Comparison', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, macs, width, label='MACs (FLOPs/2)', color='#2980b9')
    ax2.set_ylabel('MACs', color='#2980b9', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#2980b9')
    
    fig.tight_layout()
    plt.savefig('visualizations/efficiency_comparison.png', dpi=300)