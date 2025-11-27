import os
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ViTImageProcessor
from PIL import Image
from tqdm import tqdm

# Importăm modelul nostru definit anterior
# Asigură-te că fișierul fusion_model.py este în același folder (src)
from fusion_model import VeritasMultimodal

# --- CONFIGURĂRI ---
DATA_DIR = r"C:\Veritas\data"
CSV_FILE = os.path.join(DATA_DIR, "dataset_index.csv")
BATCH_SIZE = 8  # Câte știri procesează deodată (scade la 4 dacă dă eroare de memorie)
EPOCHS = 3  # De câte ori trece prin tot setul de date
LEARNING_RATE = 2e-5  # Viteza de învățare (foarte mică pentru Fine-Tuning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Antrenarea va rula pe: {DEVICE}")


# --- 1. CLASA DATASET (Cum citim datele) ---
class VeritasDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # Încărcăm tokenizerele standard
        self.tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Luăm rândul curent
        row = self.df.iloc[idx]

        # 1. Procesare Text
        text = str(row['text'])
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # 2. Procesare Imagine
        img_name = row['filename']
        folder = row['folder']  # 'real' sau 'fake'
        img_path = os.path.join(self.root_dir, folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        except Exception as e:
            # Dacă imaginea e coruptă, returnăm o imagine neagră (fallback)
            print(f"Eroare la imaginea {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
            pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values

        # 3. Label (Eticheta)
        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),  # Eliminăm dimensiunea extra [1, 128] -> [128]
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values.squeeze(0),  # [1, 3, 224, 224] -> [3, 224, 224]
            'labels': label
        }


# --- 2. LOOP-UL DE ANTRENARE ---
def train():
    # Inițializăm Dataset-ul și DataLoader-ul
    if not os.path.exists(CSV_FILE):
        print("EROARE: Nu găsesc dataset_index.csv! Rulează mai întâi 'prepare_data.py'.")
        return

    dataset = VeritasDataset(CSV_FILE, DATA_DIR)
    # Shuffle=True amestecă datele ca modelul să nu memoreze ordinea
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Inițializăm Modelul
    model = VeritasMultimodal()
    model.to(DEVICE)
    model.train()  # Punem modelul în mod "antrenare" (activează Dropout etc.)

    # Optimizator (Algoritmul care ajustează ponderile)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Funcția de pierdere (Loss function)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nÎncepe antrenarea pentru {len(dataset)} exemple...")

    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0

        progress_bar = tqdm(dataloader, desc=f"Epoca {epoch + 1}/{EPOCHS}")

        for batch in progress_bar:
            # Mutăm datele pe GPU/CPU
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # 1. Resetăm gradienții (vechi)
            optimizer.zero_grad()

            # 2. Forward Pass (Modelul face o predicție)
            outputs = model(input_ids, attention_mask, pixel_values)

            # 3. Calculăm eroarea (Loss)
            loss = criterion(outputs, labels)

            # 4. Backward Pass (Calculăm corecțiile necesare)
            loss.backward()

            # 5. Update (Aplicăm corecțiile)
            optimizer.step()

            # Statistici
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()

            # Update bară progres
            progress_bar.set_postfix({'loss': loss.item()})

        # Final de epocă
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataset)
        print(f"\nEpoca {epoch + 1} terminată. Loss Mediu: {avg_loss:.4f} | Acuratețe: {accuracy * 100:.2f}%")

    # --- SALVAREA MODELULUI ---
    save_path = "veritas_model_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel antrenat salvat cu succes în '{save_path}'!")


if __name__ == "__main__":
    train()