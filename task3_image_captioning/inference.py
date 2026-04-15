import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Assuming you'll set these up in your config.py
# MAX_LENGTH = 40
# IMAGE_SHAPE = (299, 299, 3)

def load_and_preprocess_image(image_path, target_shape=(299, 299)):
    """Loads an image and preps it for ResNet."""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_shape)
    # ResNet50 expects specific preprocessing (usually scaling to [-1, 1] or standardizing)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def get_cnn_feature_extractor():
    """Pulls ResNet50 without the classification head."""
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    # Output shape usually (None, 7, 7, 2048) -> we'll reshape in the model or here
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

def generate_caption(image_path, model, feature_extractor, vectorizer, max_length):
    """The greedy decoding loop to generate the caption."""
    # 1. Extract image features
    img = load_and_preprocess_image(image_path)
    img = tf.expand_dims(img, 0) # Add batch dimension
    img_features = feature_extractor(img)
    
    # Reshape from (1, 7, 7, 2048) to (1, 49, 2048)
    img_features = tf.reshape(img_features, (img_features.shape[0], -1, img_features.shape[3]))

    # 2. Setup vocabulary lookups
    vocab = vectorizer.get_vocabulary()
    index_to_word = tf.keras.layers.StringLookup(
        vocabulary=vocab, mask_token="", invert=True
    )

    # 3. Autoregressive Loop
    decoded_caption = "<start>"
    
    for i in range(max_length):
        # Vectorize the current generated sequence
        sequence = vectorizer([decoded_caption])
        
        # Predict the next word probabilities
        # model expects [image_features, text_sequence]
        predictions = model([img_features, sequence], training=False)
        
        # Grab the prediction for the last token in the sequence
        # predictions shape: (batch_size, sequence_length, vocab_size)
        predicted_id = tf.argmax(predictions[0, i, :], axis=-1)
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())

        if predicted_word == "<end>":
            break

        decoded_caption += " " + predicted_word

    # Clean up the output string
    final_caption = decoded_caption.replace("<start>", "").strip()
    return final_caption

def test_inference():
    """Main execution block to test a local image."""
    # Paths - adjust these once your training script outputs the weights
    IMAGE_PATH = "../data/test_image.jpg"
    MODEL_PATH = "../saved_models/transformer_captioner.keras"
    
    print("Loading ResNet feature extractor...")
    feature_extractor = get_cnn_feature_extractor()
    
    print("Loading trained Transformer model...")
    # transformer_model = tf.keras.models.load_model(MODEL_PATH)
    
    # NOTE: You will need to load the fitted TextVectorization layer here.
    # Usually, I save the vocabulary to a txt file during training, 
    # then instantiate a new TextVectorization layer and set its vocab here.
    # vectorizer = ... 

    # Example execution (uncomment when model is trained):
    # caption = generate_caption(IMAGE_PATH, transformer_model, feature_extractor, vectorizer, max_length=40)
    # print(f"\nGenerated Caption: {caption}")
    
    # # Show the image with the caption
    # img = Image.open(IMAGE_PATH)
    # plt.imshow(img)
    # plt.title(caption)
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    # test_inference()
    print("Inference script ready. Waiting for model weights to execute.")
