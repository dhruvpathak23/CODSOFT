import os

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Using absolute paths based on the project root to avoid import hell
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths (Assuming you drop Flickr8k in the data/ folder)
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "Images")          # Folder containing raw .jpg files
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt") # The CSV/TXT mapping images to text

# Save paths
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_WEIGHTS_PATH = os.path.join(SAVE_DIR, "transformer_captioner.weights.h5")
VOCAB_SAVE_PATH = os.path.join(SAVE_DIR, "vocab.pkl")

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# DATASET & TEXT PREPROCESSING
# ==========================================
# How many images to process at once. 64 is safe for 8GB VRAM. Drop to 32 if you OOM.
BATCH_SIZE = 64 

# Limit vocabulary to the top 10k most frequent words. Keeps the final dense layer manageable.
VOCAB_SIZE = 10000 

# Max words per generated caption. 40 is plenty for standard datasets like Flickr8k.
MAX_LENGTH = 40 

# Standard ResNet50 input size
IMAGE_SHAPE = (299, 299, 3) 

# ==========================================
# TRANSFORMER ARCHITECTURE
# ==========================================
# Size of the dense vector representing each word and image patch
EMBEDDING_DIM = 512 

# Transformer hidden dimension (usually matches embedding dim)
D_MODEL = 512 

# Number of hidden units in the Feed-Forward block inside the Transformer
FF_DIM = 2048 

# Number of attention heads (D_MODEL must be divisible by this)
NUM_HEADS = 8 

# ==========================================
# TRAINING PARAMS
# ==========================================
# Epochs to run. Early stopping in train.py will likely kill it before 20 anyway.
EPOCHS = 20 

# Number of images to use (helpful for quick debugging). Set to None to use full dataset.
# Set this to 1000 if you just want to test if the pipeline runs without waiting hours.
DEBUG_SUBSET_SIZE = None
