# ## Aim

# Implement a Convolutional Neural Network (CNN) to recognize
# handwritten characters using a dataset such as EMNIST. Train, evaluate,
# and display predictions on test data with confidence scores.

# ## Algorithm

# 1. Import required libraries — `idx2numpy`, NumPy, Matplotlib, TensorFlow, Keras layers.
# 2. Load EMNIST Balanced dataset, normalize pixel values, and reshape images with channel dimension.
# 3. Build a Sequential CNN with Conv2D + MaxPooling blocks, Flatten, Dense, and Dropout layers.
# 4. Compile the model with Adam optimizer and sparse categorical cross-entropy loss.
# 5. Train the model with a validation split for multiple epochs.
# 6. Evaluate the model on the test dataset to get loss and accuracy.
# 7. Plot training and validation accuracy/loss curves.
# 8. Predict a sample image's character using the mapping file and display the result with confidence.


# ## Result
# The CNN model was successfully trained on the EMNIST Balanced dataset and achieved good accuracy in classifying handwritten 
# characters across 47 classes, with the loss and accuracy curves confirming effective learning.


import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# ==========================
# Load EMNIST Dataset
# ==========================

path = r"D:\Work\Sairam\Subjects Handled\5. June-Dec 2026\Deep Learning Lab\Exp_3\archive\emnist_source_files"

x_train = idx2numpy.convert_from_file(
    path + r"\emnist-balanced-train-images-idx3-ubyte"
)

y_train = idx2numpy.convert_from_file(
    path + r"\emnist-balanced-train-labels-idx1-ubyte"
)

x_test = idx2numpy.convert_from_file(
    path + r"\emnist-balanced-test-images-idx3-ubyte"
)

y_test = idx2numpy.convert_from_file(
    path + r"\emnist-balanced-test-labels-idx1-ubyte"
)

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)


# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape)
print(x_test.shape)

#Number of classes
print("Classes:", np.unique(y_train))
print("Number of classes:", len(np.unique(y_train)))


# ==========================
# Build CNN Model
# ==========================

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(47, activation='softmax')   # EMNIST Balanced has 47 classes
])

# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=64,
    validation_split=0.2
)

# ==========================
# Evaluate Model
# ==========================

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss :", test_loss)
print("Test Accuracy :", test_accuracy)

# ==========================
# Plot Accuracy & Loss
# ==========================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


###############################################
mapping = {}

with open(r"D:\Work\Sairam\Subjects Handled\5. June-Dec 2026\Deep Learning Lab\Exp_3\archive\emnist-balanced-mapping.txt") as f:
    for line in f:
        label, ascii_code = line.split()
        mapping[int(label)] = chr(int(ascii_code))

# Select image
index = 25

# Get image and label
image = x_test[index]
label = y_test[index]

# Show image
plt.imshow(image.reshape(28, 28), cmap="gray")
#plt.title("Actual: " + mapping[label])
plt.axis("off")
plt.show()

# Predict
prediction = model.predict(np.expand_dims(image, axis=0))

predicted_label = np.argmax(prediction)

# Display result
print("Actual Character   :", mapping[label])
print("Predicted Character:", mapping[predicted_label])
print("Confidence         :", round(np.max(prediction)*100, 2), "%")