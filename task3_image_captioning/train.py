import tensorflow as tf
import os
import pickle
from src.config import BATCH_SIZE, EPOCHS, MAX_LENGTH, VOCAB_SIZE, D_MODEL, FF_DIM
from src.data_loader import get_dataset
from src.transformer import ImageCaptioningModel

def custom_loss(real, pred):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_obj(real, pred)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=loss.dtype)
    return tf.reduce_mean(loss * mask)

def custom_accuracy(real, pred):
    accuracies = tf.math.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int64))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    return tf.reduce_sum(tf.cast(tf.math.logical_and(mask, accuracies), tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))

def main():
    train_ds, val_ds, vectorizer = get_dataset() # Fixed: No arguments passed
    with open("saved_models/vocab.pkl", "wb") as f:
        pickle.dump(vectorizer.get_vocabulary(), f)

    model = ImageCaptioningModel(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH, d_model=D_MODEL, ff_dim=FF_DIM)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=custom_loss, metrics=[custom_accuracy])

    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[
        tf.keras.callbacks.ModelCheckpoint("saved_models/transformer_captioner.weights.h5", save_weights_only=True, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    ])

if __name__ == "__main__":
    main()
