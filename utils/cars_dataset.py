import pathlib
from typing import Any, Callable, Optional, Union
import scipy.io as sio
from torch.utils.data import Dataset
from PIL import Image
import os


class CustomStanfordCars(Dataset):
    """Custom Stanford Cars Dataset 
    
    Args:
        root (str or pathlib.Path): Root directory containing stanford_cars folder
        split (str): Either 'train' or 'test' 
        transform (callable, optional): Transform to apply to images
        target_transform (callable, optional): Transform to apply to targets
    """
    
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        self.root = pathlib.Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Set up paths
        self._base_folder = self.root / "stanford_cars"
        self.devkit = self._base_folder / "devkit"
        
        # Load class names
        meta_path = self.devkit / "cars_meta.mat"
        meta = sio.loadmat(str(meta_path), squeeze_me=True)
        self.classes = meta["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load samples based on split
        if split == "train":
            self._load_train_samples()
        elif split == "test":
            self._load_test_samples()
        else:
            raise ValueError(f"Split must be 'train' or 'test', got {split}")
    
    def _load_train_samples(self):
        """Load training samples with labels"""
        annos_path = self.devkit / "cars_train_annos.mat"
        images_dir = self._base_folder / "cars_train"
        
        annos = sio.loadmat(str(annos_path), squeeze_me=True)
        
        self.samples = []
        for annotation in annos["annotations"]:
            img_path = images_dir / annotation["fname"]
            # Convert 1-based class index to 0-based
            label = annotation["class"] - 1
            self.samples.append((str(img_path), label))
    
    def _load_test_samples(self):
        """Load test samples (without labels - set to -1)"""
        annos_path = self.devkit / "cars_test_annos.mat"
        images_dir = self._base_folder / "cars_test"
        
        annos = sio.loadmat(str(annos_path), squeeze_me=True)
        
        self.samples = []
        for annotation in annos["annotations"]:
            img_path = images_dir / annotation["fname"]
            # No labels available for test set
            label = -1
            self.samples.append((str(img_path), label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get image and target for given index"""
        img_path, target = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target
    
    def get_class_name(self, idx: int) -> str:
        """Get class name for given class index"""
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return "Unknown"