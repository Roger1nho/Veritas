"""
benchmark_train.py
==================
Antrenează și evaluează arhitectura multimodală (BERT-en + ViT) pe
dataset-ul FakeNewsNet (PolitiFact + GossipCop) pentru a valida că
arhitectura Veritas funcționează și pe date internaționale de referință.

Rezultatele obținute aici se compară în licență cu modelul principal
antrenat pe date românești.

Rulare:
    python benchmark_train.py
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel
from dataset import VeritasDataset  # Același dataset, alt CSV

# ── Configurare ───────────────────────────────────────────────────────────────

BENCHMARK_DIR = r"C:\Veritas\benchmark_data"
CSV_PATH      = os.path.join(BENCHMARK_DIR, "benchmark_index.csv")
MODEL_SAVE    = r"C:\Veritas\benchmark_model.pth"

BATCH_SIZE    = 4
EPOCHS        = 3
LR            = 2e-5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

class BenchmarkModel(nn.Module):
    """
    Aceeași arhitectură ca MultimodalFakeNewsModel, dar cu encodere
    în limba engleză (bert-base-uncased în loc de RoBERT românesc).
    Demonstrează că arhitectura de fuziune ponderată (gated fusion)
    funcționează indiferent de limbă.
    """

    def __init__(self, num_labels=2):
        super().__init__()

        # Encoder text – BERT engleza
        self.text_encoder  = AutoModel.from_pretrained("bert-base-uncased")
        # Encoder imagine – ViT (identic cu modelul principal)
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        text_dim  = self.text_encoder.config.hidden_size   # 768
        image_dim = self.image_encoder.config.hidden_size  # 768

        # Proiecție imagine în același spațiu ca textul
        self.image_projection = nn.Linear(image_dim, text_dim)

        # Strat de gate – calculează cât de mult contează imaginea (z ∈ [0,1])
        self.gate_layer = nn.Sequential(
            nn.Linear(text_dim + text_dim, 1),
            nn.Sigmoid()
        )

        # Clasificator final
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_emb       = self.text_encoder(input_ids=input_ids,
                                           attention_mask=attention_mask).pooler_output
        image_emb      = self.image_encoder(pixel_values=pixel_values).pooler_output
        image_emb_proj = self.image_projection(image_emb)

        # Fuziune ponderată (Gated Fusion)
        concat  = torch.cat((text_emb, image_emb_proj), dim=1)
        z       = self.gate_layer(concat)                        # ponderea imaginii
        fused   = text_emb + (z * image_emb_proj)

        logits  = self.classifier(fused)
        return logits, z   # returnăm și z pentru statistici XAI


# ── Antrenare ─────────────────────────────────────────────────────────────────

def train():
    print(f"{'='*55}")
    print(f"  VERITAS – Benchmark FakeNewsNet")
    print(f"  Device : {DEVICE.upper()}")
    print(f"{'='*55}\n")

    # Verificare fișier CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ Nu găsesc {CSV_PATH}")
        print("   Rulează prepare_benchmark.py întâi!")
        return

    # Tokenizer și procesor imagine (engleză)
    print("⏳ Încărcare modele pre-antrenate (engleză)...")
    tokenizer       = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Dataset
    full_dataset = VeritasDataset(
        csv_file        = CSV_PATH,
        root_dir        = BENCHMARK_DIR,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        max_length      = 64,   # titlurile sunt scurte, 64 tokens ajunge
        is_train        = True
    )

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=0)

    print(f"✅ Date: {len(full_dataset)} total "
          f"({len(train_ds)} train / {len(val_ds)} val)\n")

    # Model, optimizer, loss
    model     = BenchmarkModel(num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    # ── Bucla de antrenare ────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        total_loss    = 0.0
        correct_train = 0
        total_train   = 0
        z_values      = []   # colectăm ponderile imaginii pentru statistici

        loop = tqdm(train_loader, desc=f"Epoca {epoch+1}/{EPOCHS}")

        for batch in loop:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values   = batch['pixel_values'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            optimizer.zero_grad()

            logits, z = model(input_ids, attention_mask, pixel_values)
            loss      = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            preds          = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)
            z_values.extend(z.squeeze().detach().cpu().tolist())

            loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validare ──────────────────────────────────────────────────────────
        model.eval()
        correct_val = 0
        total_val   = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values   = batch['pixel_values'].to(DEVICE)
                labels         = batch['labels'].to(DEVICE)

                logits, _ = model(input_ids, attention_mask, pixel_values)
                preds     = torch.argmax(logits, dim=1)

                correct_val += (preds == labels).sum().item()
                total_val   += labels.size(0)

        train_acc = correct_train / total_train
        val_acc   = correct_val   / total_val if total_val > 0 else 0
        avg_z     = sum(z_values) / len(z_values) if z_values else 0
        avg_loss  = total_loss / len(train_loader)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "avg_image_weight": avg_z
        })

        print(
            f"📊 Epoca {epoch+1}: "
            f"Loss={avg_loss:.4f} | "
            f"Train={train_acc:.2%} | "
            f"Val={val_acc:.2%} | "
            f"Pondere imagine (z̄)={avg_z:.3f}"
        )

        # Salvăm cel mai bun model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"   💾 Model salvat (val_acc={val_acc:.2%})")

    # ── Raport final ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RAPORT FINAL – pentru licență")
    print(f"{'='*55}")
    print(f"  Dataset     : FakeNewsNet (PolitiFact + GossipCop)")
    print(f"  Arhitectură : BERT-base-uncased + ViT + Gated Fusion")
    print(f"  Epoci       : {EPOCHS}")
    print(f"  Best Val Acc: {best_val_acc:.2%}")
    print(f"\n  Detaliu pe epoci:")
    print(f"  {'Epocă':>6}  {'Loss':>7}  {'Train':>7}  {'Val':>7}  {'z̄ img':>7}")
    print(f"  {'─'*42}")
    for h in history:
        print(
            f"  {h['epoch']:>6}  "
            f"{h['loss']:>7.4f}  "
            f"{h['train_acc']:>7.2%}  "
            f"{h['val_acc']:>7.2%}  "
            f"{h['avg_image_weight']:>7.3f}"
        )
    print(f"{'='*55}")
    print(f"\n✅ Model salvat în: {MODEL_SAVE}")
    print("   Pasul următor: compară aceste rezultate cu modelul român în licență.")


if __name__ == "__main__":
    train()