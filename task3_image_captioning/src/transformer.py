import tensorflow as tf
import numpy as np

# ==========================================
# 1. POSITIONAL ENCODING & EMBEDDINGS
# ==========================================
def get_angles(pos, i, d_model):
    """Calculates the angle rates for positional encoding."""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    Generates standard sine/cosine positional encodings.
    Transformers have no sense of sequence order without this.
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    """Combines Word Embeddings with Positional Encodings."""
    def __init__(self, vocab_size, d_model, max_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_length, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # Scale the embeddings by sqrt of d_model (standard Transformer practice)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :length, :]
        return x

# ==========================================
# 2. THE TRANSFORMER DECODER BLOCK
# ==========================================
class TransformerDecoderLayer(tf.keras.layers.Layer):
    """
    A single block of the Transformer Decoder.
    Contains Masked Self-Attention, Cross-Attention with the Image, and a Feed-Forward Network.
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.seq_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, context, training=False, mask=None):
        # 1. Masked Self-Attention (Look at previously generated words)
        # We use use_causal_mask=True so the model can't cheat and look at future words
        attn1 = self.seq_attention(x, x, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1) # Residual connection

        # 2. Cross-Attention (Look at the image features)
        # Query comes from text (out1), Keys/Values come from the image (context)
        attn2 = self.cross_attention(out1, context, context)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2) # Residual connection

        # 3. Feed Forward Network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output) # Residual connection

        return out3

# ==========================================
# 3. THE FULL MODEL
# ==========================================
class ImageCaptioningModel(tf.keras.Model):
    """
    The main wrapper that glues the Vision Encoder output and the Text Decoder together.
    """
    def __init__(self, vocab_size, max_length, d_model=512, num_heads=8, ff_dim=2048):
        super().__init__()
        
        # 1. Image Feature Projection
        # ResNet outputs (batch, 49, 2048). We compress 2048 -> d_model (512) to match text embeddings.
        self.cnn_projection = tf.keras.layers.Dense(d_model, activation='relu')
        
        # 2. Text Embedding
        self.text_embedding = PositionalEmbedding(vocab_size, d_model, max_length)
        
        # 3. Transformer Decoder
        # You can stack multiple layers here, but 1 or 2 is usually enough for Flickr8k
        self.decoder_layer1 = TransformerDecoderLayer(d_model, num_heads, ff_dim)
        
        # 4. Final Output Layer
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # inputs is a tuple/list from our tf.data pipeline: (image_features, text_sequence)
        img_features, text_seq = inputs

        # Project image features to match d_model size
        context = self.cnn_projection(img_features)

        # Embed the text sequence
        x = self.text_embedding(text_seq)

        # Pass through Decoder
        x = self.decoder_layer1(x, context, training=training)

        # Output logits
        logits = self.final_layer(x)
        
        return logits
