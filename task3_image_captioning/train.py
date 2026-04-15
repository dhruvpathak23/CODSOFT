import tensorflow as tf
import os
import pickle

from src.config import BATCH_SIZE, EPOCHS, MAX_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, D_MODEL, FF_DIM
from src.data_loader import get_dataset
from src.transformer import ImageCaptioningModel

def custom_loss(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

def custom_accuracy(real, pred):
    accuracies = tf.math.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int64))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

def main():
    print("Initializing training pipeline...")

    print("Loading TF Dataset...")
    train_dataset, val_dataset, vectorizer = get_dataset(BATCH_SIZE)
    
    vocab = vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("Building the Vision-Encoder-Decoder model...")
    model = ImageCaptioningModel(
        vocab_size=VOCAB_SIZE, 
        max_length=MAX_LENGTH, 
        d_model=D_MODEL, 
        num_heads=8,
        ff_dim=FF_DIM
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=custom_loss,
        metrics=[custom_accuracy]
    )

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

    print("Starting training loop...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    print("Training complete! Model saved to /saved_models/")

if __name__ == "__main__":
    main()
