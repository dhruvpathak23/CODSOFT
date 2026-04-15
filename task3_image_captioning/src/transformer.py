import tensorflow as tf
import numpy as np

def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        return pos * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2], angle_rads[:, 1::2] = np.sin(angle_rads[:, 0::2]), np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_length, d_model)
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return x + self.pos_encoding[:, :length, :]

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation='relu'), tf.keras.layers.Dense(d_model)])
        self.layernorm1, self.layernorm2, self.layernorm3 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(3)]
        self.dropout1, self.dropout2, self.dropout3 = [tf.keras.layers.Dropout(dropout_rate) for _ in range(3)]

    def call(self, x, context, training=False):
        attn1 = self.mha1(x, x, use_causal_mask=True, training=training)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))
        attn2 = self.mha2(query=out1, value=context, key=context, training=training)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))
        return self.layernorm3(out2 + self.dropout3(self.ffn(out2), training=training))

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, max_length, d_model=512, num_heads=8, ff_dim=2048):
        super().__init__()
        self.cnn_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        self.cnn_model.trainable = False
        self.reshape = tf.keras.layers.Reshape((-1, 2048)) 
        self.cnn_projection = tf.keras.layers.Dense(d_model, activation='relu')
        self.text_embedding = PositionalEmbedding(vocab_size, d_model, max_length)
        self.decoder = TransformerDecoderLayer(d_model, num_heads, ff_dim)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        img_input, text_seq = inputs
        img_features = self.reshape(self.cnn_model(img_input, training=False))
        context = self.cnn_projection(img_features)
        return self.final_layer(self.decoder(self.text_embedding(text_seq), context, training=training))
