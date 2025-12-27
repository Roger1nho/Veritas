import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class VeritasDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, image_processor, max_length=128):
        """
        Args:
            csv_file (string): Calea către dataset_index.csv.
            root_dir (string): Calea către folderul 'data' (C:\Veritas\data).
            tokenizer: Tokenizer-ul BERT.
            image_processor: Procesorul ViT.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Extragem datele din rândul curent
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        filename = row['filename']
        folder = row['folder'] # 'real' sau 'fake'

        # 2. Construim calea completă către imagine
        # Ex: C:\Veritas\data\real\real_0.jpg
        img_path = os.path.join(self.root_dir, folder, filename)

        # 3. Procesăm Textul (BERT)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 4. Procesăm Imaginea (ViT)
        try:
            image = Image.open(img_path).convert("RGB")
            image_features = self.image_processor(images=image, return_tensors="pt")
            pixel_values = image_features['pixel_values'].squeeze()
        except Exception as e:
            print(f"⚠️ Eroare la imaginea {img_path}: {e}")
            # Dacă imaginea e coruptă, returnăm un tensor negru (fallback)
            pixel_values = torch.zeros((3, 224, 224))

        # 5. Returnăm pachetul de date (Tenzori)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }