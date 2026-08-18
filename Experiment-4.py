import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


lfw = fetch_lfw_people(min_faces_per_person=70,resize=0.5,color=False)


for i, name in enumerate(lfw.target_names):
    count = np.sum(lfw.target == i)
    print(name, ":", count, "images")

print("\nTotal images:", len(lfw.images)) 
print("Total people:", len(lfw.target_names))

X = lfw.images 
y = lfw.target

X = X / 255.0
X = np.expand_dims(X, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
    stratify=y)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(62,47,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(lfw.target_names), activation="softmax")
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

prediction = model.predict(X_test[0:1])
predicted_index = np.argmax(prediction)

print("Predicted person:", lfw.target_names[predicted_index])
print("Actual person:", lfw.target_names[y_test[0]])

plt.imshow(X_test[0].squeeze(), cmap="gray")
plt.title("Predicted: " + lfw.target_names[predicted_index])
plt.axis("off")
plt.show()

