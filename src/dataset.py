import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # <--- IMPORT NOU ESENȚIAL


class VeritasDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, image_processor, max_length=512, is_train=True):
        """
        Args:
            csv_file (string): Calea către dataset_index.csv.
            root_dir (string): Calea către folderul 'data'.
            tokenizer: Tokenizer-ul BERT.
            image_processor: Procesorul ViT.
            is_train (bool): Dacă e True, aplicăm transformări aleatoare (Data Augmentation).
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.is_train = is_train  # <--- Salvăm starea

        # --- DEFINIRE TRANSFORMATORI (Doar zgomot, fără resize) ---
        # Acestea fac modelul să nu memoreze poza exactă
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Oglindire orizontală
            transforms.RandomRotation(degrees=15),  # Rotire ușoară (+/- 15 grade)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Schimbare luminozitate/culori
            # NOTĂ: Nu facem Resize aici, se ocupă image_processor-ul ViT mai jos
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Extragem datele
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        filename = row['filename']
        folder = row['folder']

        img_path = os.path.join(self.root_dir, folder, filename)

        # 2. Procesăm Textul (BERT)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 3. Procesăm Imaginea (ViT + Augmentare)
        try:
            image = Image.open(img_path).convert("RGB")

            # --- APLICARE AUGMENTARE ---
            # Dacă suntem în timpul antrenamentului, "stricăm" puțin poza intenționat
            if self.is_train:
                image = self.augmentations(image)

            # Procesarea standard ViT (Resize la 224x224, Normalizare)
            image_features = self.image_processor(images=image, return_tensors="pt")
            pixel_values = image_features['pixel_values'].squeeze()

        except Exception as e:
            print(f"⚠️ Eroare la imaginea {img_path}: {e}")
            pixel_values = torch.zeros((3, 224, 224))

        # 4. Returnăm pachetul
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }