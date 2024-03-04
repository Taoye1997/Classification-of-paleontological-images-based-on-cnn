import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# 指定训练集和验证集目录以及图像大小
train_data_dir = r'C:\Users\11\Desktop\陶叶\增强图像\training-11.26'
val_data_dir = r'C:\Users\11\Desktop\陶叶\增强图像\valid-11.26'  # 替换为你的验证集目录
img_size = (224, 224)

# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 从目录加载训练数据
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical'  # 如果是分类任务
)

# 数据增强
val_datagen = ImageDataGenerator(rescale=1./255)

# 从目录加载验证数据
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=img_size,
    batch_size=32,
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

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 定义回调函数
checkpoint = ModelCheckpoint('vgg16_zengqiang_weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# 训练模型
history = model.fit(train_generator, epochs=100, validation_data=val_generator, callbacks=[checkpoint])

# 保存模型
model.save('vgg16_success_11.23')

# 获取损失和准确度的历史数据
loss = history.history['loss']
accuracy = history.history['accuracy']

# 获取验证损失和准确度的历史数据
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# 绘制准确率和损失函数曲线
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失函数曲线
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

