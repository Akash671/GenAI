import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Step 1: Load Pre-trained Model and Tokenizer
model_name = "bert-base-uncased"  # Replace with your model of choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

# Step 2: Load Predefined Dataset (e.g., IMDb reviews)
dataset = load_dataset("imdb")

# Step 3: Preprocess Data
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Convert datasets to TensorFlow format
def convert_to_tf_dataset(tokenized_dataset):
    features = {key: tokenized_dataset[key] for key in ["input_ids", "attention_mask"]}
    labels = tokenized_dataset["label"]
    return tf.data.Dataset.from_tensor_slices((features, labels))

train_dataset = convert_to_tf_dataset(tokenized_datasets["train"]).shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)
test_dataset = convert_to_tf_dataset(tokenized_datasets["test"]).batch(16).prefetch(tf.data.AUTOTUNE)

# Step 4: Compile the Model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Step 5: Train the Model
model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# Step 6: Evaluate the Model
results = model.evaluate(test_dataset)
print("Evaluation results:", results)

# Optional: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_tf_model")
tokenizer.save_pretrained("./fine_tuned_tf_model")

