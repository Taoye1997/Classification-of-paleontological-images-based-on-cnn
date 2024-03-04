import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import warnings

# 设置随机种子，以保证结果的可重复性
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=r'C:\Users\11\Desktop\陶叶\Original\training', transform=transform)
# 用于验证的数据集
val_dataset = datasets.ImageFolder(root=r'C:\Users\11\Desktop\陶叶\Original\validation', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 忽略torchvision的模型加载相关的警告
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")

# 加载预训练的 ResNet18 模型
model = models.resnet18(pretrained=True)  # 使用预训练的权重
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 11)  # 假设有11个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练和验证模型
num_epochs =100
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    # 训练模型
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100.0 * correct / total
    train_acc_history.append(train_accuracy)
    train_loss_history.append(running_loss / len(train_loader))

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100.0 * correct / total
    val_acc_history.append(val_accuracy)
    val_loss_history.append(val_running_loss / len(val_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%, '
          f'Validation Loss: {val_running_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# 绘制准确率曲线图
plt.subplot(2, 1, 1)
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# 绘制损失函数曲线图
plt.subplot(2, 1, 2)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss', color='red')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


