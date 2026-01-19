from pathlib import Path

class Config: 
    # Paths
    DATA_DIR = Path("data/rgb_d_dataset")
    OUTPUT_DIR = Path("outputs")
    FEATURES_DIR = OUTPUT_DIR / "features"
    MODELS_DIR = OUTPUT_DIR / "models"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    
    # Dataset
    NUM_POSITIONS = 35
    TRAIN_SESSIONS = [0, 1, 2]
    TEST_SESSIONS = [3, 4]
    IMAGE_SIZE = 224
    
    # Models to benchmark
    MODELS = {
        "ResNet50": {"pretrained": True, "feature_dim": 2048},
        "EfficientNet_B0": {"pretrained": True, "feature_dim": 1280},
        "ViT_B16": {"pretrained": True, "feature_dim": 768},
        "DenseNet121": {"pretrained": True, "feature_dim": 1024},
    }
    
    # Training
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "cuda"
    
    # Classification
    CLASSIFIER = "KNN"
    K_NEIGHBORS = 5
