import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Load the VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet', include_top=False)

# Select the specified convolutional layers
layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
layers = [base_model.get_layer(name).output for name in layer_names]

# Create a new model that outputs the selected layers' activations
activation_model = Model(inputs=base_model.input, outputs=layers)

# Path to a sample image
sample_image_path = r'C:\Users\11\Desktop\陶叶\碳酸盐岩颗粒11.13\Carbon\all\oolite\044_resized.jpg'

# Load the sample image
img = image.load_img(sample_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get the feature maps for the sample image
activations = activation_model.predict(img_array)

# Visualize the feature maps for the specified layers
for i, activation in enumerate(activations):
    num_features = activation.shape[-1]
    size = activation.shape[1]

    # Number of columns in the display grid
    cols = num_features // 8
    display_grid = np.zeros((size, cols * size))

    # Post-process the feature to be visually palatable
    for col in range(cols):
        channel_image = activation[0, :, :, col * 8]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[:, col * size : (col + 1) * size] = channel_image

    # Display the grid
    scale = 1.0 / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_names[i])
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
