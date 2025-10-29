import torch
import torchvision.transforms as transforms
from PIL import Image, ImageStat
import numpy as np
import warnings
import io

# warnings.filterwarnings('ignore')


#                ANALIZOR IMAGINE – VERITAS
#  Imaginile manipulate sau de calitate slabă trădează adesea mici defecte. Acest modul încearcă să le "vâneze" prin analiza unor caracteristici cheie.

def analyze_image(image_path):
    """
    Funcția principală care orchestrează analiza unei imagini.
    Primește calea către o imagine și returnează un scor de suspiciune.
    """
    print(f"Analizez imaginea de la: {image_path}")
    try:
        # Deschidem imaginea in format RGB
        image = Image.open(image_path).convert('RGB')
        print(f"Imagine încărcată cu succes ({image.size[0]}x{image.size[1]} pixeli). Extragerea trasaturilor")
    except Exception as e:
        print(f"A apărut o eroare la încărcarea imaginii: {e}")
        # Returnăm un scor neutru daca imaginea nu poate fi citita.
        return 0.5

    # Standardizam dimensiunea imaginii pentru a ne asigura că analiza este consistentă, indiferent de rezoluția originală.
    # Apoi o convertim într-un format numeric (Tensor) cu care PyTorch poate lucra.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)

    # Extragem o serie de indicii numerice din imagine.
    features = extract_image_clues(img_tensor, image)

    # Pe baza indiciilor adunate, calculăm un scor final de probabilitate.
    suspicion_score = calculate_suspicion_score(features)

    return suspicion_score


def extract_image_clues(img_tensor, original_image):
    """
    Extrage un set de "indicii" numerice din imagine, care ne ajută să-i
    evaluăm autenticitatea și calitatea.
    """
    print("\nCaut indicii relevante în imagine:")

    features = {
        'noise_level': torch.std(img_tensor).item(),
        'brightness': torch.mean(img_tensor).item(),
        'contrast': torch.var(img_tensor).item(),
        'saturation': compute_saturation(img_tensor),
        'edge_clarity': compute_edge_strength(img_tensor),
        'sharpness': estimate_sharpness(original_image),
        'jpeg_artifacts': estimate_jpeg_artifacts(original_image)
    }
    for name, value in features.items():
        print(f"   - {name.replace('_', ' ').capitalize():<20}: {value:.4f}")

    return features


def compute_edge_strength(img_tensor):
    """Calculează cât de pronunțate sunt marginile obiectelor din imagine."""
    diff_x = img_tensor[:, :, 1:] - img_tensor[:, :, :-1]
    diff_y = img_tensor[:, 1:, :] - img_tensor[:, :-1, :]
    return float(torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))) / 2


def estimate_sharpness(image):
    """Estimează claritatea generală (focusul) imaginii."""
    gray = image.convert("L")  # Analiza claritații e mai simplă pe alb-negru.
    arr = np.array(gray, dtype=np.float32)
    gy, gx = np.gradient(arr)
    variance = np.var(np.sqrt(gx ** 2 + gy ** 2))
    return float(variance / 10000)  # Normalizam valoarea pentru a fi mai ușor de interpretat.


def estimate_jpeg_artifacts(image):
    """
    Simulează o recompresie JPEG pentru a vedea cât de mult se degradează imaginea.
    Dacă se degradează puțin, înseamnă că era deja foarte comprimată.
    """
    buffer = io.BytesIO()
    # Salvam o copie a imaginii în memorie la o calitate scăzută (50).
    image.save(buffer, format='JPEG', quality=50)
    recompressed = Image.open(buffer).convert('RGB')

    # Măsuram diferenta medie de pixeli intre original si copia degradata.
    diff = np.mean(np.abs(np.array(image, dtype=np.float32) - np.array(recompressed, dtype=np.float32)))
    return float(diff / 255.0)  # Normalizăm la o valoare între 0 și 1.


def compute_saturation(img_tensor):
    """Măsoară intensitatea și vivacitatea culorilor."""
    r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
    # O imagine saturată are diferențe mari între canalele de culoare (roșu, verde, albastru).
    return torch.mean(torch.abs(r - g) + torch.abs(r - b) + torch.abs(g - b)).item() / 3


def calculate_suspicion_score(features):
    """
    Combină toate indiciile adunate folosind un model logistic simplu.
    Fiecare indiciu are o anumită "greutate" în decizia finală.
    """
    # Convertim dicționarul de trăsături într-un vector numeric.
    # Ordinea este importantă și trebuie să corespundă cu cea a ponderilor!
    x = np.array([
        features['noise_level'],
        features['brightness'],
        features['contrast'],
        features['saturation'],
        features['edge_clarity'],
        features['sharpness'],
        features['jpeg_artifacts']
    ])

    # Aceste ponderi au fost calibrate empiric. De exemplu, 'jpeg_artifacts' (1.4) cântărește mai mult în decizie decât 'brightness' (0.8).
    weights = np.array([1.2, 0.8, 0.6, 1.1, 0.9, 0.7, 1.4])
    bias = -1.8  # Un prag de pornire.

    # Calculăm scorul brut, combinând indiciile cu ponderile lor.
    z = np.dot(x, weights) + bias
    # Transformăm scorul brut într-o probabilitate (între 0 și 1) folosind funcția sigmoidă.
    probability = 1 / (1 + np.exp(-z))

    # Ne asigurăm că scorul nu atinge extremele (0 sau 1), pentru a reflecta incertitudinea.
    return float(max(0.05, min(0.95, probability)))

def print_detailed_analysis(score):
    """Prezintă rezultatul final într-un mod ușor de înțeles."""
    print("\n" + "=" * 50)
    print("REZULTATUL ANALIZEI IMAGINII")
    print("=" * 50)
    print(f"\nSCOR DE SUSPICIUNE: {score:.2f} (adică o probabilitate de {score * 100:.2f}%)")

    if score > 0.75:
        verdict = "RISC RIDICAT: Imaginea prezintă multiple semne de manipulare sau calitate foarte slabă."
    elif score > 0.6:
        verdict = "ATENȚIE: S-au detectat câteva anomalii. Imaginea ar putea fi editată sau scoasă din context."
    elif score > 0.4:
        verdict = "NEUTRU: Analiza nu a putut trage o concluzie clară. Nu există semne evidente de manipulare."
    else:
        verdict = "PROBABIL AUTENTICA: Imaginea pare să fie de bună calitate și fără artefacte suspecte."

    print(f"VERDICT: {verdict}")
    print("=" * 50)

if __name__ == "__main__":
    test_image_path = "C:\\Users\\QUASAR\\Desktop\\Veritas\\images\\ART-1073-1.jpg"

    print("Inițiez analizorul de imagini Veritas...")
    print("-" * 55)

    try:
        final_score = analyze_image(test_image_path)
        print_detailed_analysis(final_score)
    except FileNotFoundError:
        print(
            f"EROARE: Fișierul nu a fost găsit la calea specificată. Verifică dacă '{test_image_path}' este corectă.")
    except Exception as e:
        print(f"O eroare neașteptată a avut loc: {e}")