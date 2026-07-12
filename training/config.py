DATASET_ID  = "marmal88/skin_cancer"
LABEL_FIELD = "dx"
VAL_SPLIT   = 0.15
SEED        = 42

HAM_LABELS  = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis-like_lesions",
    "dermatofibroma",
    "melanocytic_Nevi",
    "melanoma",
    "vascular_lesions",
]
NUM_CLASSES = len(HAM_LABELS)

MODEL_BASE_ID = "facebook/dinov2-base"
HUB_REPO      = "Ganymede981/ham10000-vit"


LORA_R       = 32
LORA_ALPHA   = 64         
LORA_DROPOUT = 0.05
HEAD_DROPOUT = 0.1

IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.7630, 0.5456, 0.5700]
NORMALIZE_STD  = [0.1409, 0.1521, 0.1697]


EPOCHS               = 60
BATCH_SIZE_TRAIN     = 32
BATCH_SIZE_EVAL      = 64
GRAD_ACCUM_STEPS     = 2
LEARNING_RATE        = 3e-4
LR_SCHEDULER         = "cosine"
WARMUP_RATIO         = 0.10
WEIGHT_DECAY         = 0.05
LABEL_SMOOTHING      = 0.05
EARLY_STOPPING_PAT   = 8
METRIC_FOR_BEST      = "f1_macro"


CUTMIX_ALPHA = 1.0
MIXUP_ALPHA  = 0.4

OUTPUT_DIR        = "checkpoints"
FINAL_MODEL_DIR   = "final_model"
