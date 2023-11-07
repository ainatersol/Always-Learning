from PIL import Image
import os
import glob
import pandas as pd
import numpy as np
import re
from torchvision import transforms


from torch.utils.data import Dataset


# Build dataset and dataloader
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, meta, range_start, range_end, transform=None, augmentations=None):
        self.image_paths = glob.glob(f"{img_dir}*-cropped.jpg")
        #         self.ids = [ re.search(r'(.+)-cropped', os.path.basename(p)).group(1) for p in self.image_paths]
        self.ids, self.image_paths = zip(
            *[
                (re.search(r"(.+)-cropped", os.path.basename(p)).group(1), p)
                for p in self.image_paths
                if re.search(r"(.+)-cropped", os.path.basename(p)).group(1) in list(meta["image_id"])
            ]
        )

        self.ids = self.ids[range_start:range_end]
        self.image_paths = self.image_paths[range_start:range_end]
        self.transform = transform
        self.augmentations = augmentations

        ids_df = pd.DataFrame({"image_id": self.ids})
        self.labels_df = ids_df.merge(meta[["image_id", "modifiediou_score", "ai_accuracy"]])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("L"))
        image_id = self.ids[idx]

        image_label = self.labels_df["modifiediou_score"][self.labels_df["image_id"] == image_id].iloc[0]

        if self.augmentations is not None:
            samples = self.augmentations(image=image)
            image = samples['image']

        if self.transform is not None:
            image = self.transform(image)

        return image, image_label

class CustomImageDatasetLarge(Dataset):
    def __init__(self, img_dir, meta, range_start, range_end, transform=None, augmentations=None):
        self.image_paths = glob.glob(f"{img_dir}*-cropped.jpg")
        #         self.ids = [ re.search(r'(.+)-cropped', os.path.basename(p)).group(1) for p in self.image_paths]
        self.ids, self.cropped_image_paths, self.large_image_paths = zip(
            *[
                (
                    re.search(r"(.+)-cropped", os.path.basename(p)).group(1), 
                    p, 
                    p.replace('-cropped', '')
                )
                for p in self.image_paths
                if re.search(r"(.+)-cropped", os.path.basename(p)).group(1) in list(meta["image_id"])
            ]
        )

        self.ids = self.ids[range_start:range_end]
        self.cropped_image_paths = self.cropped_image_paths[range_start:range_end]
        self.large_image_paths = self.large_image_paths[range_start:range_end]
        self.transform = transform
        self.augmentations = augmentations

        ids_df = pd.DataFrame({"image_id": self.ids})
        self.labels_df = ids_df.merge(meta[["image_id", "modifiediou_score", "ai_accuracy"]])

    def __len__(self):
        return len(self.large_image_paths)

    def __getitem__(self, idx):
        crop_image = Image.open(self.cropped_image_paths[idx]).convert("L")
        large_image = Image.open(self.large_image_paths[idx]).convert("L")
        image_id = self.ids[idx]

        image_label = self.labels_df["modifiediou_score"][self.labels_df["image_id"] == image_id].iloc[0]

        transform2 = transforms.Compose([
                transforms.Resize((256, 256)),
            ])  
        crop_image_np = np.array(transform2(crop_image))
        large_image_np = np.array(transform2(large_image))
        image = np.dstack((crop_image_np, large_image_np, large_image_np))

        if self.augmentations:
            samples = self.augmentations(image=image)
            image = samples['image']

        if self.transform:
            image = self.transform(image)

        return image, image_label
    
def create_data(train_dataset):
    data_train = [(sample[0].numpy().reshape(-1), sample[1]) for sample in train_dataset]
    X_train, y_true_train = zip(*data_train)
    X_train = np.array(X_train)
    y_true_train = np.array(y_true_train) 
    y_true_train_binary = np.where(y_true_train > 0.5, 1.0, 0.0)
    return X_train, y_true_train, y_true_train_binary