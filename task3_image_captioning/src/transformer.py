import tensorflow as tf
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_length, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :length, :]
        return x

class TransformerDecoderLayer(tf.keras.layers.Layer):
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

    def call(self, x, context, training=False):
        # x is the text sequence, context is the flattened image features
        attn1 = self.seq_attention(x, x, use_causal_mask=True, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1) 

        # Cross-attention: Query from text, Key/Value from image context
        attn2 = self.cross_attention(query=out1, value=context, key=context, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2) 

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output) 

        return out3

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, max_length, d_model=512, num_heads=8, ff_dim=2048):
        super().__init__()
        # NEW: Encoder to extract and flatten CNN features
        self.cnn_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        self.cnn_model.trainable = False
        
        # Flatten spatial dimensions (7x7 -> 49)
        self.reshape = tf.keras.layers.Reshape((-1, 2048)) 
        self.cnn_projection = tf.keras.layers.Dense(d_model, activation='relu')
        
        self.text_embedding = PositionalEmbedding(vocab_size, d_model, max_length)
        self.decoder_layer = TransformerDecoderLayer(d_model, num_heads, ff_dim)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        img_input, text_seq = inputs
        
        # Process Image
        img_features = self.cnn_model(img_input, training=False)
        img_features = self.reshape(img_features) # (batch, 49, 2048)
        context = self.cnn_projection(img_features) # (batch, 49, d_model)
        
        # Process Text
        x = self.text_embedding(text_seq)
        
        # Decode
        x = self.decoder_layer(x, context, training=training)
        
        # Final logits
        logits = self.final_layer(x)
        return logits
