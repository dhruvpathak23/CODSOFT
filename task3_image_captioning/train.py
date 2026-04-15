import tensorflow as tf
import os
import pickle

# --- HYPERPARAMETERS ---
# In a real setup, move these to src/config.py
BATCH_SIZE = 64
EPOCHS = 20
MAX_LENGTH = 40
VOCAB_SIZE = 10000
EMBEDDING_DIM = 512
D_MODEL = 512
FF_DIM = 2048

# --- IMPORTS (Placeholder for your src modules) ---
# from src.data_loader import get_dataset, get_vectorizer
# from src.transformer import ImageCaptioningModel

def custom_loss(real, pred):
    """
    Standard SparseCategoricalCrossentropy, but we mask the 0s (padding).
    This stops the model from optimizing for padding tokens.
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(real, pred)
    
    # Create a mask to ignore padding (assuming 0 is the pad token)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    
    loss *= mask
    return tf.reduce_mean(loss)

def custom_accuracy(real, pred):
    """
    Accuracy metric that also ignores padding tokens.
    """
    accuracies = tf.math.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int64))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

def main():
    print("Initializing training pipeline...")

    # 1. Load Data (You'll implement this in data_loader.py)
    # dataset should yield: ( (image_features, input_text_sequence), target_text_sequence )
    # Input sequence:  "<start> a dog is running"
    # Target sequence: "a dog is running <end>"
    # print("Loading TF Dataset...")
    # train_dataset, val_dataset = get_dataset(BATCH_SIZE)
    
    # 2. Setup Text Vectorizer and save Vocab for inference
    # vectorizer = get_vectorizer(VOCAB_SIZE, MAX_LENGTH)
    # vocab = vectorizer.get_vocabulary()
    # print(f"Vocabulary size: {len(vocab)}")
    
    # Save the vocab so inference.py can load it
    # os.makedirs("saved_models", exist_ok=True)
    # with open("saved_models/vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)

    # 3. Instantiate the Model
    print("Building the Vision-Encoder-Decoder model...")
    # model = ImageCaptioningModel(
    #     vocab_size=VOCAB_SIZE, 
    #     max_length=MAX_LENGTH, 
    #     d_model=D_MODEL, 
    #     ff_dim=FF_DIM
    # )

    # 4. Compile the Model
    # We use Adam with a slightly lower learning rate to keep the attention weights stable early on.
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    #     loss=custom_loss,
    #     metrics=[custom_accuracy]
    # )

    # 5. Callbacks
    # Save the best model weights so you don't lose everything if it OOMs or crashes
    checkpoint_path = "saved_models/transformer_captioner.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_loss"
    )

    # 6. Train
    print("Starting training loop...")
    # history = model.fit(
    #     train_dataset,
    #     epochs=EPOCHS,
    #     validation_data=val_dataset,
    #     callbacks=[checkpoint_callback, early_stopping]
    # )
    
    print("Training complete! Model saved to /saved_models/")

if __name__ == "__main__":
    main()
