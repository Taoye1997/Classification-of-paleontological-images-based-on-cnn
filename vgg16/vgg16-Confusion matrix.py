import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 指定测试集目录和图像大小
test_data_dir = r'C:\Users\11\Desktop\陶叶\增强图像\test-11.26'  # 替换为你的测试集目录
img_size = (224, 224)

# 数据增强
test_datagen = ImageDataGenerator(rescale=1./255)

# 从目录加载测试数据
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=1,
    shuffle=False,  # 保持文件顺序以便混淆矩阵的正确性
    class_mode='categorical'
)

num_classes = 11  # 你的类别数目，需要根据你的任务设定

# 加载预训练的VGG16模型，不包含顶层（全连接层）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结预训练模型的权重
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义顶层
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)

# 在定义 `predictions` 之前使用 `x`
predictions = layers.Dense(num_classes, activation='softmax')(x)

# 构建新模型
model = models.Model(inputs=base_model.input, outputs=predictions)

# 加载之前训练好的模型权重
model.load_weights(r'D:\python\pythonProject\vgg16_sucess_11.23\vgg16_zengqiang_weights.h5')

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 在测试集上进行预测
y_true_labels = test_generator.classes
y_pred_probabilities = model.predict(test_generator)
y_pred_labels = tf.argmax(y_pred_probabilities, axis=1)

# 计算混淆矩阵和分类报告
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
classification_rep = classification_report(y_true_labels, y_pred_labels)

# 将混淆矩阵中的数值转换为百分比
conf_matrix_percent = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True) * 100

# 打印混淆矩阵和分类报告
print("Confusion Matrix (Counts):")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# 打印百分比形式的混淆矩阵
print("\nConfusion Matrix (Percentage):")
print(conf_matrix_percent)

# 绘制混淆矩阵图
class_names = [str(i) for i in range(num_classes)]
sns.heatmap(conf_matrix_percent, annot=True, fmt=".1f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Percentage)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
