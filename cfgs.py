import logging
import os

# =====================================================
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =====================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'data')

IMAGE_FOLDER = os.path.join(DATASET_ROOT, 'plane_sky')
OUTPUT_PATH = os.path.join(ROOT, 'logs')
CKPT_PATH = os.path.join(OUTPUT_PATH, 'ckpts')
RESULT_PATH = os.path.join(OUTPUT_PATH, 'results')

# =====================================================
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# =====================================================
BATCH_SIZE = 1
IMG_HEIGHT = 240
IMG_WIDTH = 240
CHANNEL = 3
MAX_ITERATION = 1000
