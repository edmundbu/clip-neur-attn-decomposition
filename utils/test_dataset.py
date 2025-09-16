import os

from torch.utils.data import Dataset
from PIL import Image

TEST_DIR = 'test_data/test'


def reduce_to_5000_files(files_dir: str) -> None:
    all_files = sorted(os.listdir(files_dir))
    for idx, file in enumerate(all_files):
        if idx % 20 != 0:
            os.remove(os.path.join(files_dir, file))
            print(f"removed {idx}")


class ImageNetTest(Dataset):
    def __init__(self, folder, transform):
        self.files = sorted(os.listdir(folder))
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.files[idx])
        image = Image.open(file_path).convert("RGB")  
        image = self.transform(image)
        return image


if __name__ == "__main__":
    reduce_to_5000_files(files_dir=TEST_DIR)