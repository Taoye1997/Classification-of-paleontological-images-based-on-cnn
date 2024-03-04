import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
test_dataset = datasets.ImageFolder(root=r'C:\Users\11\Desktop\陶叶\增强图像\test-11.26', transform=transform)

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的 ResNet18 模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(test_dataset.classes))  # 类别数量为数据集的类别数量

# 加载模型权重
model.load_state_dict(torch.load(r'D:\python\pythonProject\resnet_model\variables\resnet18_model_weights.pth'))
model.eval()

# 用于存储真实标签和预测标签的列表
all_labels = []
all_predictions = []

# 遍历测试集，获取真实标签和预测标签
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions, normalize='true')  # 使用normalize='true'将数值转为百分比

# 将混淆矩阵数据转为DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=test_dataset.classes, columns=test_dataset.classes)

# 显示混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap="Blues")  # 使用fmt=".2%"将显示的数值格式化为百分比
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 打印混淆矩阵数据
print("Confusion Matrix:\n", conf_matrix_df)

# 显示分类报告
print("Classification Report:\n", classification_report(all_labels, all_predictions, target_names=test_dataset.classes))
# 显示混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap="Blues")

# 在图上添加混淆矩阵数据
for i in range(len(test_dataset.classes)):
    for j in range(len(test_dataset.classes)):
        plt.text(j + 0.5, i + 0.5, f"{conf_matrix[i, j]:.2%}", ha='center', va='center', color='red')

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
