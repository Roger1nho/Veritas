from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

def analyze_text(text):
    """
    Analizează un text și returnează un scor de încredere că este fake news.

    Args:
        text (str): Textul știrii de analizat.

    Returns:
        float: Scorul de încredere (0-1) că știrea este falsă.
    """

    model_name = "dumitrescustefan/bert-base-romanian-uncased-v1"

    print("Se incarca modelul de text...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    fake_confidence = probabilities[0][1].item()

    return fake_confidence

if __name__ == '__main__':
    test_text="Conform datelor oficiale publicate de Institutul Național de Statistică, economia României a crescut cu 4,5% în trimestrul al treilea față de aceeași perioadă a anului trecut, potrivit comunicatului de presă emis de instituție."
    print("Analizator de Fake News")
    print(f"Textul care este testat: '{test_text}'")

    confidence = analyze_text(test_text)
    print(f"Fake score: {confidence:.4f} ({confidence*100:.2f})")

    if confidence > 0.6:
        print("Acest articol nu pare de incredere")
    elif confidence > 0.4:
        print("Este neclar nivelul de integritate al acestui articol")
    else:
        print("Acest articol pare de incredere")
