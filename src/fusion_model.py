import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, ViTModel, ViTConfig

from transformers import AutoTokenizer, ViTImageProcessor
from PIL import Image
import requests


class VeritasMultimodal(nn.Module):
    def __init__(self):
        super(VeritasMultimodal, self).__init__()

        # 1. RAMURA TEXT (BERT Românesc)
        # Folosim AutoModel (fără cap de clasificare) pentru a extrage doar trăsăturile (embeddings)
        self.text_model_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)

        # 2. RAMURA IMAGINE (Vision Transformer - ViT)
        # Folosim un ViT pre-antrenat de la Google
        self.image_model_name = "google/vit-base-patch16-224-in21k"
        self.image_encoder = ViTModel.from_pretrained(self.image_model_name)

        # Dimensiunile output-urilor (Hidden sizes)
        # BERT base are de obicei 768, ViT base are tot 768
        self.text_hidden_size = self.text_encoder.config.hidden_size
        self.image_hidden_size = self.image_encoder.config.hidden_size

        # 3. COMPONENTA DE FUZIUNE (Concatenare + Clasificare)
        fusion_input_size = self.text_hidden_size + self.image_hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Prevenim overfitting-ul
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Ieșire: 2 clase (Real vs Fake)
            # Nu punem Softmax aici dacă folosim CrossEntropyLoss la antrenare
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        input_ids, attention_mask: Tensors de la tokenizer-ul BERT
        pixel_values: Tensorul imaginii procesat de feature extractor-ul ViT
        """

        # --- Procesare Text ---
        # Trecem textul prin BERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Luăm vectorul [CLS] care reprezintă tot textul (primul token)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # --- Procesare Imagine ---
        # Trecem imaginea prin ViT
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        # La ViT, luăm tot vectorul [CLS] (similar cu BERT)
        image_features = image_outputs.last_hidden_state[:, 0, :]

        # --- Fuziune ---
        # Concatenăm vectorii (îi lipim unul lângă altul)
        combined_features = torch.cat((text_features, image_features), dim=1)

        # --- Clasificare ---
        logits = self.classifier(combined_features)

        return logits


# --- Exemplu de utilizare (Inference / Antrenare) ---

if __name__ == "__main__":
    # Initializare Tokenizer si Procesor de Imagine
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Initializare Model Veritas
    model = VeritasMultimodal()

    # Date demo
    text_input = "Acesta este un articol suspect despre economie care pare fals."
    # O imagine random de pe net pentru test
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image_input = Image.open(requests.get(url, stream=True).raw)

    # 1. Pregătire date (Preprocessing)
    text_inputs = tokenizer(text_input, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    image_inputs = image_processor(images=image_input, return_tensors="pt")

    # 2. Rulare model (Forward pass)
    # Modelul va returna logits (scoruri brute).
    # Notă: Rezultatul va fi random până când antrenezi modelul!
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=image_inputs['pixel_values']
        )

    # 3. Interpretare
    probs = torch.nn.functional.softmax(outputs, dim=1)
    fake_prob = probs[0][1].item()  # Presupunem că indexul 1 e clasa "Fake"

    print(f"Scoruri brute (Logits): {outputs}")
    print(f"Probabilitate FAKE: {fake_prob:.4f}")