import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer, ViTImageProcessor

# Importăm clasele noastre
from dataset import VeritasDataset
from fusion_model import MultimodalFakeNewsModel

# --- CONFIGURĂRI ---
BASE_DIR = r"C:\Veritas\data"
CSV_PATH = os.path.join(BASE_DIR, "dataset_index.csv")
MODEL_SAVE_PATH = "veritas_model.pth"

BATCH_SIZE = 4  # Mai mic dacă ai erori de memorie
EPOCHS = 10  # De câte ori trecem prin date (fiind date puține, 10 e ok)
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    print(f"🚀 Pornire antrenare pe: {DEVICE.upper()}")

    # 1. Încărcare Tokenizer și Procesor
    print("⏳ Încărcare modele pre-antrenate...")
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # 2. Încărcare Dataset
    full_dataset = VeritasDataset(
        csv_file=CSV_PATH,
        root_dir=BASE_DIR,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # Împărțim datele: 80% antrenare, 20% validare
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"✅ Date încărcate: {len(full_dataset)} total ({len(train_dataset)} Train, {len(val_dataset)} Val)")

    # 3. Inițializare Model
    model = MultimodalFakeNewsModel(num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()

    # 4. Bucla de Antrenare
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        loop = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{EPOCHS}")

        for batch in loop:
            # Mutăm pe GPU
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Forward
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calcul acuratețe antrenare
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            loop.set_postfix(loss=loss.item())

        # 5. Validare (Testăm pe datele nevăzute)
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids, attention_mask, pixel_values)
                preds = torch.argmax(outputs, dim=1)

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        train_acc = correct_train / total_train
        val_acc = correct_val / total_val if total_val > 0 else 0

        print(
            f"📊 Epoca {epoch + 1}: Loss={total_loss / len(train_loader):.4f} | Train Acc={train_acc:.2%} | Val Acc={val_acc:.2%}")

    # 6. Salvare
    print(f"💾 Salvare model în {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("🎉 Antrenare completă!")


if __name__ == "__main__":
    train()