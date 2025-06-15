import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from timm.data import Mixup
import wandb


class SoftCrossEntropyLoss(torch.nn.Module):
    
    def __init__(self, label_smoothing=0.0, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        num_classes = outputs.size(-1)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
        
        if targets.dim() == 1:  # Hard labels
            # Convert to one-hot and apply label smoothing
            targets_one_hot = torch.zeros_like(outputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            if self.label_smoothing > 0:
                # Apply label smoothing: (1-α) * one_hot + α/K
                smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                               self.label_smoothing / num_classes
            else:
                smooth_targets = targets_one_hot
                
        else:  # Soft labels
            smooth_targets = targets
            # Optionally apply additional smoothing to soft labels
            if self.label_smoothing > 0:
                uniform_dist = torch.ones_like(targets) / num_classes
                smooth_targets = (1 - self.label_smoothing) * targets + \
                               self.label_smoothing * uniform_dist
        
        # Compute cross entropy with soft targets
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, num_channels) # Squeeze
        y = self.fc(y).view(batch_size, num_channels, 1, 1) # Excitation
        return x * y.expand_as(x)

class DeeperCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        # Block 1
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.batchnorm1 = torch.nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.drop2d_1 = torch.nn.Dropout2d(p=0.2)
        
        # Block 2
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.batchnorm2 = torch.nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.drop2d_2 = torch.nn.Dropout2d(p=0.2)
        
        # Block 3
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.batchnorm3 = torch.nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2) 
        self.drop2d_3 = torch.nn.Dropout2d(p=0.2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(256, 512)
        self.dropout = torch.nn.Dropout(0.5) 
        self.fc2 = torch.nn.Linear(512, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.batchnorm1(self.conv1(x)))
        x = self.se1(x)
        x = self.pool1(x)
        x = self.drop2d_1(x)
        x = torch.nn.functional.gelu(self.batchnorm2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool2(x)
        x = self.drop2d_2(x)
        x = torch.nn.functional.gelu(self.batchnorm3(self.conv3(x)))
        x = self.se3(x)
        x = self.pool3(x)
        x = self.drop2d_3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Main execution block ---
if __name__ == '__main__':
    # --- Parameters & Configuration ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = {
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 64,
        "optimizer": "AdamW",
        "scheduler_step": 5,
        "scheduler_gamma": 0.1,
        "architecture": "CustomCNN",
        "dataset": "CIFAR-10",
        "validation_split": 0.2,
        "max_lr": 0.01,
        "mixup_alpha": 0.4,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "mixup_prob": 0.8,
        "switch_prob": 0.5,
    }

    wandb.init(
        project="image-classification-cifar10",
        name="deeper-cnn-cifar10-mps-cutmix-50-epochs",
        config=config
    )
    config = wandb.config

    # --- Data Loading and Transforms for CIFAR-10 ---
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    val_size = int(len(full_train_dataset) * config.validation_split)
    train_size = len(full_train_dataset) - val_size
    train_set, val_set = random_split(full_train_dataset, [train_size, val_size])
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
    num_classes = len(full_train_dataset.classes)

    # --- Training Setup ---
    model = DeeperCNN(num_classes=num_classes).to(device)
    train_criterion = SoftCrossEntropyLoss(label_smoothing=config.label_smoothing)
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=config["max_lr"],
                                                epochs=config["epochs"],
                                                steps_per_epoch=len(train_loader))
    mixup_fn = Mixup(
        mixup_alpha=config["mixup_alpha"],
        cutmix_alpha=config["cutmix_alpha"],
        prob=config["mixup_prob"],
        switch_prob=config["switch_prob"],
        num_classes=num_classes
    )

    wandb.watch(model, train_criterion, log="all", log_freq=100)

    # --- Training Loop ---
    best_val_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = mixup_fn(inputs, labels)
            outputs = model(inputs)
            loss = train_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = val_criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / train_size
        val_loss = val_loss / val_size
        val_acc = val_corrects.float() / val_size
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        wandb.log({
            "Epoch": epoch,
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "Accuracy/val": val_acc,
            "Learning_rate": current_lr
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            

    wandb.finish()
    print(f"Training Complete. Best validation accuracy: {best_val_acc:.4f}")