import numpy as np
import tensorflow as tf
from keras import layers, models
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def load_labeled_data(file_path):
    with open(file_path, 'rb') as file:
        labeled_data = pickle.load(file)
    return labeled_data

loaded_labeled_data = load_labeled_data("./model/model3/labeled_data.pkl")

# Extract features and labels from the loaded data
X = np.array([features for features, label in loaded_labeled_data])
y = np.array([label for features, label in loaded_labeled_data])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))

# Assuming the shape of features is (num_samples, image_height, image_width, channels)
# image_height, image_width, channels = X.shape[1:]
image_height, image_width, channels = 480, 640, 3

# Split the data into training, validation, and test sets
train_val_images, test_images, train_val_labels, test_labels = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_val_images, train_val_labels, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Calculate the total number of elements in the original array
original_size = train_images.size

# Calculate the total number of elements required for the new shape
new_size = image_height * image_width * channels * len(train_images)

# Check if the sizes match
print(original_size)
print(new_size)
assert original_size == new_size, "Sizes do not match!"


# Reshape input data to have the correct shape
train_images = train_images.reshape(-1, image_height, image_width, channels)
val_images = val_images.reshape(-1, image_height, image_width, channels)
test_images = test_images.reshape(-1, image_height, image_width, channels)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Adjust output layer units based on the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")