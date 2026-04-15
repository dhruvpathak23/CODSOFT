import tensorflow as tf
import os
import string
from src.config import BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE, IMAGE_SHAPE, IMAGE_DIR, CAPTION_FILE, DEBUG_SUBSET_SIZE

def load_captions_data(caption_file, image_dir):
    """
    Parses the text file containing image-to-caption mappings.
    Expects standard format: image_name.jpg, caption text
    """
    image_paths = []
    captions = []
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        # Skip header if it exists
        lines = f.readlines()[1:] if "image" in f.readline() else f.readlines()
        
        for line in lines:
            # Handle standard comma-separated lines
            parts = line.strip().split(',', 1)
            if len(parts) < 2:
                continue
                
            img_name, caption = parts[0], parts[1]
            full_img_path = os.path.join(image_dir, img_name)
            
            # Standardize text: lowercase, remove punctuation, add <start> and <end> tokens
            caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
            caption = f"<start> {caption} <end>"
            
            image_paths.append(full_img_path)
            captions.append(caption)

    if DEBUG_SUBSET_SIZE:
        print(f"DEBUG MODE: Truncating dataset to {DEBUG_SUBSET_SIZE} samples.")
        image_paths = image_paths[:DEBUG_SUBSET_SIZE]
        captions = captions[:DEBUG_SUBSET_SIZE]

    return image_paths, captions

def custom_standardization(input_string):
    """
    We need a custom standardization function because standard Keras TextVectorization 
    strips all punctuation, including the < and > in our <start> and <end> tokens.
    """
    lowercase = tf.strings.lower(input_string)
    # Strip basic punctuation but leave < and >
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(string.punctuation.replace('<', '').replace('>', '')), "")

def get_vectorizer(captions, vocab_size, max_length):
    """
    Sets up the TextVectorization layer and builds the vocabulary.
    """
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_length,
        standardize=custom_standardization
    )
    # Fit the vocabulary to the captions
    vectorizer.adapt(captions)
    return vectorizer

def load_image(image_path, caption):
    """
    Loads and preprocesses a single image.
    Used within the tf.data pipeline.
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:2])
    # ResNet50 specific preprocessing
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, caption

def process_dataset(image, caption, vectorizer):
    """
    Prepares the input and target sequences for Teacher Forcing.
    """
    # Vectorize the text
    vectorized_caption = vectorizer(caption)
    
    # Input sequence: <start> a dog is running
    # Target sequence: a dog is running <end>
    # We slice to shift the targets by one timestep
    input_seq = vectorized_caption[:-1]
    target_seq = vectorized_caption[1:]
    
    return (image, input_seq), target_seq

def build_tf_dataset(image_paths, captions, vectorizer, is_training=True):
    """
    Assembles the highly optimized tf.data pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Load images in parallel
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Vectorize text and prepare (input, target) structure
    dataset = dataset.map(
        lambda img, cap: process_dataset(img, cap, vectorizer), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch for GPU efficiency
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def get_dataset(batch_size=BATCH_SIZE):
    """
    Main entry point for train.py to pull the dataset.
    """
    import re # Imported here for the custom_standardization regex
    
    print("Loading raw text data...")
    image_paths, captions = load_captions_data(CAPTION_FILE, IMAGE_DIR)
    
    # Simple train/val split (80/20)
    split_idx = int(len(image_paths) * 0.8)
    train_paths, val_paths = image_paths[:split_idx], image_paths[split_idx:]
    train_captions, val_captions = captions[:split_idx], captions[split_idx:]
    
    print("Building vocabulary...")
    vectorizer = get_vectorizer(train_captions, VOCAB_SIZE, MAX_LENGTH)
    
    print("Assembling TF Data pipelines...")
    train_dataset = build_tf_dataset(train_paths, train_captions, vectorizer, is_training=True)
    val_dataset = build_tf_dataset(val_paths, val_captions, vectorizer, is_training=False)
    
    return train_dataset, val_dataset, vectorizer
