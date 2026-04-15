import os

# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt") 

SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_WEIGHTS_PATH = os.path.join(SAVE_DIR, "transformer_captioner.weights.h5")
VOCAB_SAVE_PATH = os.path.join(SAVE_DIR, "vocab.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 64 
VOCAB_SIZE = 10000 
MAX_LENGTH = 40 
IMAGE_SHAPE = (299, 299, 3) 

EMBEDDING_DIM = 512 
D_MODEL = 512 
FF_DIM = 2048 
NUM_HEADS = 8 
EPOCHS = 20
