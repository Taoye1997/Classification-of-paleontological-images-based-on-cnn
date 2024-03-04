import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Model  # Import Model class
import seaborn as sns

# Load VGG16 model
model = VGG16(weights='imagenet')

# Path to the sample image
sample_image_path = r"C:\Users\11\Desktop\陶叶\增强图像\热力图\c.jpg"

# Load the sample image and preprocess it
img = image.load_img(sample_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get the intermediate layer's output
layer_name = 'block5_conv3'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(img_array)

# Get the predicted label
decoded_predictions = decode_predictions(model.predict(img_array), top=3)[0]
predicted_label = decoded_predictions[0][1]

# Plot the heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(intermediate_output[0, :, :, 0], cmap='viridis')

# Set the title and axis labels
plt.title(f"Heatmap for {predicted_label}")
plt.xlabel('Width')
plt.ylabel('Height')

# Show the plot
plt.show()
