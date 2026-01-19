import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
from config import Config
from data_loader import RGBDDataset
from models import FeatureExtractorFactory

class FeatureExtractor:
    """Pipeline to extract visual features from entire dataset."""
    
    def __init__(self, model_name, device=Config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = FeatureExtractorFactory.get_extractor(model_name, device)
        Config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    def extract_features(self, sessions):
        """Extract features for specified sessions."""
        dataset = RGBDDataset(
            Config.DATA_DIR,
            image_size=Config.IMAGE_SIZE,
            sessions=sessions
        )
        
        if len(dataset) == 0:
            print(f"No images found for sessions {sessions}")
            return None
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            shuffle=False
        )
        
        all_features = []
        all_positions = []
        
        with torch.no_grad():
            for images, positions in tqdm(dataloader, desc=f"Extracting {self.model_name}"):
                images = images.permute(0, 3, 1, 2).to(self.device)
                features = self.model(images)
                features = features.squeeze().cpu().numpy()
                
                if features.ndim == 1:
                    features = features[np.newaxis, :]
                
                all_features.append(features)
                all_positions.extend(positions.numpy())
        
        return np.vstack(all_features), np.array(all_positions)
    
    def save_features(self, features, positions, split_name):
        """Save features in HDF5 format."""
        output_path = Config.FEATURES_DIR / f"{self.model_name}_{split_name}.h5"
        
        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=features, compression="gzip")
            f.create_dataset("positions", data=positions, compression="gzip")
        
        print(f"Features saved: {output_path}")