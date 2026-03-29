import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel


class MultimodalFakeNewsModel(nn.Module):
    def __init__(self, num_labels=2):
        super(MultimodalFakeNewsModel, self).__init__()

        #Modul Text: BERT Românesc
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")

        #Modul Imagine: Vision Transformer (ViT)
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        #Dimensiuni
        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.config.hidden_size

        self.image_projection = nn.Linear(image_dim, text_dim)

        self.gate_layer = nn.Sequential(
            nn.Linear(text_dim + text_dim, 1),
            nn.Sigmoid()
        )

        #Clasificator Final (Fuziune)
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Previne memorarea mecanică
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Features
        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_emb = self.image_encoder(pixel_values=pixel_values).pooler_output
        image_emb_proj = self.image_projection(image_emb)

        concat_features = torch.cat((text_emb, image_emb_proj), dim=1)
        z = self.gate_layer(concat_features) # <-- ACEASTA ESTE PONDEREA IMAGINII

        # Fuziune Ponderată
        fused_embedding = text_emb + (z * image_emb_proj)
        logits = self.classifier(fused_embedding)

        # Returnăm atât predicția, cât și 'z' pentru statistici XAI
        return logits, z