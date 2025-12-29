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

        self.image_projection = nn.Linear(image_dim, text_dim)

        self.gate_layer = nn.Sequential(
            nn.Linear(text_dim + text_dim, 1),
            nn.Sigmoid()
        )

        # 4. Clasificator Final (Fuziune)
        # Combinăm vectorii (768 + 768 = 1536)
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Previne memorarea mecanică
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # 1. Features
        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_emb = self.image_encoder(pixel_values=pixel_values).pooler_output

        # 2. Aliniem dimensiunile (Image -> Text Space)
        image_emb_proj = self.image_projection(image_emb)

        # 3. Calculăm "Gating Weight" (z)
        # Ne uităm la ambele și decidem cât contează imaginea în contextul acestui text
        concat_features = torch.cat((text_emb, image_emb_proj), dim=1)
        z = self.gate_layer(concat_features) # z va fi un scalar per exemplu (ex: 0.2)

        # 4. Fuziune Ponderată
        # Formula: Final = Text + (z * Image)
        # Dacă z e mic, imaginea e ignorată. Dacă z e mare, imaginea influențează decizia.
        fused_embedding = text_emb + (z * image_emb_proj)

        logits = self.classifier(fused_embedding)

        return logits