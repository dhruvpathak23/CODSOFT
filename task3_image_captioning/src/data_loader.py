import tensorflow as tf
import os
import string
import re  
from src.config import BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE, IMAGE_SHAPE, IMAGE_DIR, CAPTION_FILE

def load_captions_data(caption_file, image_dir):
    image_paths, captions = [], []
    with open(caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:] if "image" in f.readline() else f.readlines()
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) < 3:
                parts = line.strip().split(',', 1)
                if len(parts) < 2: continue
                img_name, caption = parts[0], parts[1]
            else:
                img_name, caption = parts[0], parts[2]
            
            full_img_path = os.path.join(image_dir, img_name)
            caption = f"<start> {caption.lower().translate(str.maketrans('', '', string.punctuation))} <end>"
            image_paths.append(full_img_path)
            captions.append(caption)
    return image_paths, captions

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(string.punctuation.replace('<', '').replace('>', '')), "")

def get_vectorizer(captions, vocab_size, max_length):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, output_mode="int",
        output_sequence_length=max_length, standardize=custom_standardization
    )
    vectorizer.adapt(captions)
    return vectorizer

def load_image(image_path, caption):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:2])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, caption

def process_dataset(image, caption, vectorizer):
    vectorized_caption = vectorizer(caption)
    return (image, vectorized_caption[:-1]), vectorized_caption[1:]

def build_tf_dataset(image_paths, captions, vectorizer, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
    if is_training: dataset = dataset.shuffle(1000)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda img, cap: process_dataset(img, cap, vectorizer), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def get_dataset():
    image_paths, captions = load_captions_data(CAPTION_FILE, IMAGE_DIR)
    split = int(len(image_paths) * 0.8)
    vectorizer = get_vectorizer(captions[:split], VOCAB_SIZE, MAX_LENGTH)
    train_ds = build_tf_dataset(image_paths[:split], captions[:split], vectorizer)
    val_ds = build_tf_dataset(image_paths[split:], captions[split:], vectorizer, is_training=False)
    return train_ds, val_ds, vectorizer
