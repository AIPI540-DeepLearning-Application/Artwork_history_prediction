import os
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image

'''
    This is for data augmentation
'''

class ArtworkDataHandler:
    def __init__(self, train_data_path, val_data_path, train_save_dir, val_save_dir, image_height=224, image_width=224, batch_size=32):
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size

        # Paths
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_save_dir = train_save_dir
        self.val_save_dir = val_save_dir

        # Create directories for saving images
        os.makedirs(train_save_dir, exist_ok=True)
        os.makedirs(val_save_dir, exist_ok=True)

        # Initialize transformations
        self.init_transforms()

        # Load datasets and create data loaders
        self.train_loader, self.val_loader = self.create_data_loaders()

    def init_transforms(self):
        # Training transformations
        self.train_transforms = [
            transforms.Compose([transforms.Resize((self.image_height, self.image_width)), transforms.ToTensor()]),
            transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
            transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]),
            transforms.Compose([transforms.RandomResizedCrop((self.image_height, self.image_width), scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ToTensor()]),
            transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor()])
        ]

        # Validation transformations
        self.val_transforms = [
            transforms.Compose([transforms.Resize((self.image_height, self.image_width)), transforms.ToTensor()]),
            transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
            transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]),
            transforms.Compose([transforms.RandomResizedCrop((self.image_height, self.image_width), scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ToTensor()]),
            transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor()])
        ]

    def create_data_loaders(self):
        # Training datasets
        train_datasets = [datasets.ImageFolder(root=self.train_data_path, transform=transform) for transform in self.train_transforms]
        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation datasets
        val_datasets = [datasets.ImageFolder(root=self.val_data_path, transform=transform) for transform in self.val_transforms]
        val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def save_dataset(self, dataset, save_dir, prefix='image'):
        for i, (image, label) in enumerate(dataset):
            image_path = os.path.join(save_dir, f"{prefix}_{i}_label_{label}.jpg")
            save_image(image, image_path)

# Usage
train_data_path = '../data/train/'
val_data_path = '../data/val/'
train_save_dir = '../data/train_tf'
val_save_dir = '../data/val_tf'

data_handler = ArtworkDataHandler(train_data_path, val_data_path, train_save_dir, val_save_dir)

# Using the data loaders
# for images, labels in data_handler.train_loader:

# Saving transformed images
data_handler.save_dataset(data_handler.train_loader.dataset, data_handler.train_save_dir, prefix='train')
data_handler.save_dataset(data_handler.val_loader.dataset, data_handler.val_save_dir, prefix='val')
