"""
prepare_benchmark_figshare.py
==============================
Pregătește benchmark_index.csv din dataset-ul Figshare (fakeddit_subset).

Structura așteptată:
    C:\\Veritas\\figshare\\fakeddit_subset\\
        image_folder\\          <- imagini training
        validation_image\\      <- imagini validare
        training_data_fakeddit.jsonl
        validation_data_fakeddit.jsonl

Output:
    C:\\Veritas\\benchmark_data\\real\\*.jpg
    C:\\Veritas\\benchmark_data\\fake\\*.jpg
    C:\\Veritas\\benchmark_data\\benchmark_index.csv

Rulare:
    python prepare_benchmark_figshare.py
"""

import os
import json
import shutil
import re
from pathlib import Path

# ── Configurare — schimbă dacă ai dezarhivat altundeva ───────────────────────

FIGSHARE_DIR  = r"C:\Veritas\figshare\fakeddit_subset"
BENCHMARK_DIR = r"C:\Veritas\benchmark_data"
MAX_PER_CLASS = 2000   # 4000 real + 4000 fake = 8000 total

# ── Creare foldere output ─────────────────────────────────────────────────────

os.makedirs(os.path.join(BENCHMARK_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(BENCHMARK_DIR, "fake"), exist_ok=True)


def extract_filename(file_uri):
    """
    Extrage numele fișierului din URI de tipul:
    gs://my_trial_bucket_finetune/image_folder/9a46c1362ec06f0ffbd2578fa777ea8d.jpg
    """
    return Path(file_uri).name


def parse_jsonl(jsonl_path, image_dir):
    """
    Parsează un fișier JSONL și returnează lista de înregistrări:
    [{"text": ..., "label": 0/1, "image_path": ...}, ...]
    """
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                contents = obj.get("contents", [])

                # Extragem textul și imaginea din mesajul user
                user_parts = contents[0]["parts"]
                model_answer = contents[1]["parts"][0]["text"].strip()

                # Labelul: "Yes" = fake (1), "No" = real (0)
                label = 1 if model_answer == "Yes" else 0

                # Extragem titlul din text
                text_part = next(
                    (p["text"] for p in user_parts if "text" in p), ""
                )
                # Titlul e între ghilimele după "Title:"
                match = re.search(r'Title:"([^"]+)"', text_part)
                title = match.group(1) if match else text_part[:100]

                # Extragem numele imaginii
                file_part = next(
                    (p for p in user_parts if "fileData" in p), None
                )
                if not file_part:
                    continue

                img_filename = extract_filename(
                    file_part["fileData"]["fileUri"]
                )
                img_path = os.path.join(image_dir, img_filename)

                if not os.path.exists(img_path):
                    continue   # sărind dacă imaginea nu există local

                records.append({
                    "text":       title,
                    "label":      label,
                    "image_path": img_path,
                    "img_filename": img_filename,
                })

            except Exception:
                continue

    return records


def main():
    print("=" * 55)
    print("  VERITAS – Benchmark Figshare (Fakeddit subset)")
    print("=" * 55)

    # Căi fișiere
    train_jsonl  = os.path.join(FIGSHARE_DIR, "training_data_fakeddit.jsonl")
    val_jsonl    = os.path.join(FIGSHARE_DIR, "validation_data_fakeddit.jsonl")
    train_imgdir = os.path.join(FIGSHARE_DIR, "image_folder")
    val_imgdir   = os.path.join(FIGSHARE_DIR, "validation_image")

    # Verificare
    for p in [train_jsonl, val_jsonl, train_imgdir, val_imgdir]:
        if not os.path.exists(p):
            print(f"❌ Nu găsesc: {p}")
            print(f"   Verifică că FIGSHARE_DIR e setat corect.")
            return

    # Parsare ambele fișiere JSONL
    print("\n📂 Parsare JSONL-uri...")
    all_records = []
    all_records += parse_jsonl(train_jsonl, train_imgdir)
    all_records += parse_jsonl(val_jsonl,   val_imgdir)

    real_all = [r for r in all_records if r["label"] == 0]
    fake_all = [r for r in all_records if r["label"] == 1]

    print(f"   Total parsate : {len(all_records)}")
    print(f"   Real          : {len(real_all)}")
    print(f"   Fake          : {len(fake_all)}")

    # Selectăm MAX_PER_CLASS din fiecare
    real_selected = real_all[:MAX_PER_CLASS]
    fake_selected = fake_all[:MAX_PER_CLASS]

    print(f"\n✅ Selectate pentru benchmark:")
    print(f"   Real : {len(real_selected)}")
    print(f"   Fake : {len(fake_selected)}")

    # Copiem imaginile și construim index
    import pandas as pd
    index_records = []

    print(f"\n📋 Copiere imagini...")

    for label_name, subset, label_int in [
        ("real", real_selected, 0),
        ("fake", fake_selected, 1)
    ]:
        print(f"  [{label_name.upper()}]")
        for i, rec in enumerate(subset, 1):
            dst_filename = f"fakeddit_{label_name}_{i}.jpg"
            dst_path     = os.path.join(BENCHMARK_DIR, label_name, dst_filename)

            try:
                shutil.copy2(rec["image_path"], dst_path)
                status = "✅"
            except Exception as e:
                print(f"    ❌ Eroare la {rec['img_filename']}: {e}")
                status = "❌"
                continue

            if i % 50 == 0 or i == len(subset):
                print(f"    {status} {i}/{len(subset)} copiate")

            index_records.append({
                "text":     rec["text"],
                "filename": dst_filename,
                "folder":   label_name,
                "label":    label_int,
            })

    # Salvare CSV
    df_out   = pd.DataFrame(index_records)
    out_path = os.path.join(BENCHMARK_DIR, "benchmark_index.csv")
    df_out.to_csv(out_path, index=False)

    real_count = (df_out["label"] == 0).sum()
    fake_count = (df_out["label"] == 1).sum()

    print(f"\n{'='*55}")
    print(f"  GATA!")
    print(f"  Total          : {len(df_out)}")
    print(f"  Real           : {real_count}")
    print(f"  Fake           : {fake_count}")
    print(f"  Index salvat în: {out_path}")
    print(f"{'='*55}")
    print("\nPasul următor: rulează benchmark_train.py")


if __name__ == "__main__":
    main()