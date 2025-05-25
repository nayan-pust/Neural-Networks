import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt


from google.colab import drive
drive.mount('/content/drive')

# Set paths to your dataset

fruits_path = '/content/drive/MyDrive/FruitsImage'

# Create image dataset for the training dataset
batch_size = 32
img_size = (100, 100)  # Make sure this size is consistent with your model

train_ds = image_dataset_from_directory(
    fruits_path,
    label_mode='int',  # 'int' to assign labels based on folder names
    image_size=img_size,
    batch_size=batch_size
)

# Get the number of classes from dataset
class_names = train_ds.class_names
num_classes = len(class_names)

# Define CNN architecture
model = models.Sequential([
    layers.InputLayer(input_shape=(100, 100, 3)),  # Update input shape to match image size

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set training options
epochs = 20

# Train the CNN
history = model.fit(
    train_ds,
    epochs=epochs
)

# Save the trained model
model_save_path ='/content/drive/MyDrive/fruits_model.h5'
model.save(model_save_path)

# Load and preprocess a random sample image for classification
sample_image_file = random.choice(train_ds.file_paths)
sample_image = tf.keras.preprocessing.image.load_img(sample_image_file, target_size=img_size)
input_image = tf.keras.preprocessing.image.img_to_array(sample_image)
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Classify the sample image
predicted_label = model.predict(input_image)
predicted_class = class_names[np.argmax(predicted_label)]

# Display the predicted label and the sample image
print(f'Predicted label: {predicted_class}')

plt.imshow(sample_image)
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()