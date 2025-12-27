import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel


class MultimodalFakeNewsModel(nn.Module):
    def __init__(self, num_labels=2):
        super(MultimodalFakeNewsModel, self).__init__()

        # 1. Modul Text: BERT Românesc
        self.text_encoder = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

        # 2. Modul Imagine: Vision Transformer (ViT)
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # 3. Dimensiuni
        text_dim = self.text_encoder.config.hidden_size  # 768
        image_dim = self.image_encoder.config.hidden_size  # 768

        # 4. Clasificator Final (Fuziune)
        # Combinăm vectorii (768 + 768 = 1536)
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Previne memorarea mecanică
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # A. Extragem trăsături TEXT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_outputs.pooler_output  # Reprezentarea frazei (Vector 768)

        # B. Extragem trăsături IMAGINE
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_emb = image_outputs.pooler_output  # Reprezentarea imaginii (Vector 768)

        # C. CONCATENARE (Fuziune)
        combined_features = torch.cat((text_emb, image_emb), dim=1)

        # D. CLASIFICARE (Real vs Fake)
        logits = self.classifier(combined_features)

        return logits