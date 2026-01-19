import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import cv2

class RGBDDataset(Dataset):
    """Load RGB-D images from indoor localization dataset."""
    
    def __init__(self, data_dir, image_size=224, sessions=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.sessions = sessions or []
        self.images = []
        self.positions = []
        self._build_dataset()
    
    def _build_dataset(self):
        """Build index of images and associated positions."""
        for session in self.sessions:
            session_path = self.data_dir / f"session_{session}"
            if not session_path.exists():
                continue
            
            for position_id in range(35):
                position_dir = session_path / f"position_{position_id}"
                if position_dir.exists():
                    rgb_files = sorted(position_dir.glob("*_rgb.png"))
                    for img_file in rgb_files:
                        self.images.append(str(img_file))
                        self.positions.append(position_id)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img, self.positions[idx]