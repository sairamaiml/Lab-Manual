import tensorflow as tf
import numpy as np

#Loading corpus
file_path = r"D:\Work\Sairam\Subjects Handled\5. June-Dec 2026\Deep Learning Lab\Exp_4\shakespeare.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Corpus loaded successfully.")
print("Corpus length:", len(text))
print(text)

print("\nFirst 10 characters:")
print(text[:10])


#Create Character Vocabulary

vocab = sorted(set(text))
vocab_size = len(vocab)
#print("Characters:", vocab)

# Character to number
char_to_int = {char: i for i, char in enumerate(vocab)}
print(char_to_int)
print("Vocabulary size:", vocab_size)

# Convert text to numbers
text_as_int = np.array([char_to_int[char] for char in text])
print(text_as_int)

print("Sample text:", text[:5])
print("Integer representation:", text_as_int[:5])

#Create Input and Target Sequences
sequence_length = 100

X = []
Y = []

for i in range(len(text_as_int) - sequence_length):
    X.append(
        text_as_int[i:i + sequence_length]
    )

    Y.append(
        text_as_int[i + 1:i + sequence_length + 1]
    )

X = np.array(X)
Y = np.array(Y)

print("Input shape:", X.shape)
print("Target shape:", Y.shape)

#Create TensorFlow Dataset
batch_size = 64

dataset = tf.data.Dataset.from_tensor_slices((X, Y))

dataset = dataset.shuffle(10000).batch(batch_size,drop_remainder=True)

print("Dataset created successfully")
print("Batch size:", batch_size)
#############################################

import numpy as np

print("NumPy version:", np.__version__)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=65,
        output_dim=128
    ),

    tf.keras.layers.SimpleRNN(
        256,
        return_sequences=True
    ),

    tf.keras.layers.Dense(
        65
    )
])

X = tf.random.uniform(
    shape=(2, 5),
    minval=0,
    maxval=65,
    dtype=tf.int32
)

output = model(X)

print("Output shape:", output.shape)
########################################################

#Compile and train the Model
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam",loss=loss_function)
history = model.fit(dataset,epochs=1)


# Integer to Character and generate text
int_to_char = {
    i: char for i, char in enumerate(vocab)
}


def generate_text(start_text, length=200):

    result = start_text

    for i in range(length):

        # Convert last 100 characters to numbers
        x = [
            char_to_int[c]
            for c in result[-100:]
        ]

        # Convert to NumPy array
        x = np.array([x])

        # Predict next character
        prediction = model.predict(
            x,
            verbose=0
        )

        # Get the predicted character ID
        next_id = np.argmax(
            prediction[0, -1]
        )

        # Convert character ID back to character
        next_char = int_to_char[next_id]

        # Add the predicted character
        result += next_char

    return result

# Generate Text


text = generate_text(
    "Shall ",
    200
)

print("\nGenerated Text:")
print(text)