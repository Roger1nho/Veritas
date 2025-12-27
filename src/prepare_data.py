import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
from io import BytesIO
import time

# --- CONFIGURARE ---
BASE_DIR = r"C:\Veritas\data"
os.makedirs(os.path.join(BASE_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "fake"), exist_ok=True)

# LISTA DE LINK-URI PENTRU ANTRENARE
# (Adaugă aici cât mai multe linkuri către articole specifice)

urls_real = [
    "https://www.digisport.ro/special/habar-n-are-nimic-nu-stie-cum-l-a-numit-gigi-becali-pe-nicusor-dan-de-craciun-hai-sa-ti-spun-3999701",
    "https://www.digi24.ro/stiri/externe/rusia/rusia-ar-putea-amplasa-noi-rachete-hipersonice-la-o-fosta-baza-aeriana-din-belarus-extinde-raza-de-actiune-in-europa-3562777",
    "https://www.digi24.ro/stiri/actualitate/furtuna-pe-valea-prahovei-un-copac-a-cazut-peste-o-masina-cu-trei-persoane-la-sinaia-din-cauza-vantului-puternic-3562685",
    "https://www.digi24.ro/stiri/economie/2026-anul-de-calibrare-pentru-2027-irineu-darau-despre-misiunea-guvernului-sa-punem-caramizile-viitoarei-dezvoltari-3562595",
    "https://recorder.ro/stirile-zilei/23-decembrie-2025-daniel-david-a-plecat-problemele-din-educatie-raman/",
    "https://recorder.ro/stirile-zilei/22-decembrie-2025-consultari-in-justitie-sistemul-judiciar-in-moarte-clinica/",
    "https://recorder.ro/stirile-zilei/19-decembrie-2025-presedinta-curtii-de-apel-si-realitatea-paralela/",
    "https://recorder.ro/stirile-zilei/18-decembrie-2025-panica-printre-boti-cum-e-combatut-documentarul-justitie-capturata/",
    "https://recorder.ro/stirile-zilei/17-decembrie-2025-sistemul-contraataca-primele-represalii-la-adresa-judecatoarei-morosanu/",
    "https://recorder.ro/stirile-zilei/16-decembrie-2025-justitie-capturata-sesizare-oficiala-pe-numele-sefului-dna/",
    "https://recorder.ro/stirile-zilei/15-decembrie-2025-decizie-cedo-scut-pentru-judecatoarea-morosanu/",
    "https://recorder.ro/stirile-zilei/12-decembrie-2025-curajul-e-contagios-aproape-700-de-magistrati-reclama-justitia-capturata/",
    "https://recorder.ro/stirile-zilei/11-decembrie-2025-ziua-in-care-justitia-romana-a-explodat/",
    "http://recorder.ro/stirile-zilei/10-decembrie-2025-justitie-capturata-pensiile-speciale-rezista/",
    "https://recorder.ro/stirile-zilei/9-decembrie-2025-nicusor-dan-joaca-la-extrema/",
    "https://recorder.ro/stirile-zilei/8-decembrie-2025-alegeri-partiale-ciucu-si-ciolacu-viitorul-coalitiei/",
    "https://recorder.ro/stirile-zilei/5-decembrie-2025-criza-apei-furnizarea-se-reia-cu-taraita/",
    "https://recorder.ro/stirile-zilei/4-decembrie-2025-alegerile-din-bucuresti-aur-ridica-miza/",
    "https://recorder.ro/stirile-zilei/3-decembrie-2025-cum-s-a-facut-noroi-apa-de-baut-coruptie-si-incompetenta/",
    "https://recorder.ro/stirile-zilei/2-decembrie-2025-cursa-pentru-pmb-inghesuiala-ca-n-tramvai/",
    "https://recorder.ro/stirile-zilei/unirea-in-ura-sarbatorita-separat/",
    "https://recorder.ro/stirile-zilei/28-noiembrie-2025-la-loc-comanda-ionut-mosteanu-pleaca-de-la-mapn/",
    "https://recorder.ro/stirile-zilei/28-noiembrie-2025-la-loc-comanda-ionut-mosteanu-pleaca-de-la-mapn/",
    "https://recorder.ro/stirile-zilei/27-noiembrie-2025-ionut-mosteanu-probleme-in-cv-si-risc-de-evacuare-din-guvern/",
    "https://recorder.ro/stirile-zilei/26-noiembrie-2025-fraudele-euroins-anchetate-dupa-trei-ani/",
    "https://recorder.ro/stirile-zilei/25-noiembrie-2025-atentie-cad-drone-romania-si-moldova-survolate/",
    "https://recorder.ro/stirile-zilei/25-noiembrie-2025-atentie-cad-drone-romania-si-moldova-survolate/",
    "https://recorder.ro/stirile-zilei/24-noiembrie-2025-turul-doi-chiar-nu-mai-vine-inapoi-revin-insa-serviciile-in-strategia-de-aparare/",
    "https://recorder.ro/stirile-zilei/21-noiembrie-2025-anca-alexandrescu-emisiuni-de-milioane-de-lei-din-buget/",
    "https://recorder.ro/stirile-zilei/20-noiembrie-2025-intoarcerea-mercenarului-potra-repatriat/",
    "https://recorder.ro/stirile-zilei/19-noiembrie-2025-pensiile-magistratilor-runda-a-doua/",
    "https://recorder.ro/stirile-zilei/18-noiembrie-2025-ludovic-orban-presedintele-joaca-la-granita-constitutiei/",
    "https://recorder.ro/stirile-zilei/17-noiembrie-2025-neglijenta-letala-aproape-douazeci-de-politisti-cercetati/",
    "https://recorder.ro/stirile-zilei/14-noiembrie-2025-drum-peste-magistralele-de-gaz-masuri-si-plangeri-penale/",
    "https://recorder.ro/stirile-zilei/13-noiembrie-2025-pensiile-magistratilor-sa-se-revizuiasca-dar-sa-nu-se-schimbe-mai-nimic/",
    "https://recorder.ro/stirile-zilei/12-noiembrie-2025-legea-nordis-adoptata-cu-unanimitate/",
    "https://recorder.ro/stirile-zilei/11-noiembrie-2025-administratia-incaiera-asociatia-conflict-la-varful-daruieste-viata/",
    "https://recorder.ro/stirile-zilei/10-noiembrie-2025-cincizeci-si-una-de-crime-nicio-demisie/",
    "https://recorder.ro/stirile-zilei/7-noiembrie-2025-alegere-grea-in-psd-grindeanu-versus-grindeanu/",
    "https://recorder.ro/stirile-zilei/6-noiembrie-2025-dosarul-de-coruptie-din-vaslui-primele-retineri/",
    "https://recorder.ro/stirile-zilei/5-noiembrie-2025-reteaua-rezervistilor-cine-ar-fi-planuit-sa-l-scoata-pe-georgescu-din-tara/",
    "https://recorder.ro/stirile-zilei/4-noiembrie-2025-ploua-cu-eliberari-azi-un-fugar-si-doi-fosti-ministri/",
    "https://recorder.ro/stirile-zilei/3-noiembrie-2025-georgescu-trage-presul-de-sub-picioarele-suveranistilor/",
    "https://recorder.ro/stirile-zilei/31-octombrie-2025-comemorarea-colectiv-plansul-peste-program-amendat/",
    "https://recorder.ro/stirile-zilei/30-octombrie-2025-zece-ani-de-la-colectiv-simti-ca-ti-a-murit-copilul-degeaba/",
    "https://recorder.ro/stirile-zilei/29-octombrie-2025-reducere-nu-retragere-de-ce-pleaca-sute-de-soldati-americani-din-romania/",
    "https://recorder.ro/stirile-zilei/28-octombrie-2025-inca-o-moarte-suspecta-in-spital-directoarea-medicala-a-sj-buzau/",
    "https://recorder.ro/stirile-zilei/27-octombrie-2025-pe-locuri-fiti-gata-start-spre-pmb/",
    "https://recorder.ro/stirile-zilei/24-octombrie-2025-nicusor-dan-femeile-nu-sunt-protejate-de-stat/",
    "https://recorder.ro/stirile-zilei/23-octombrie-2025-toti-murim-raspunsul-cinic-al-primarului-negoita-la-panica-cetatenilor-din-s3/",
    "https://recorder.ro/stirile-zilei/22-octombrie-2025-asii-amanetului-fac-bani-cu-o-campanie-antidrog/",
    "http://recorder.ro/stirile-zilei/21-octombrie-2025-procurorii-extind-ancheta-nordis/"
]

urls_fake = [
    "https://www.timesnewroman.ro/monden/un-hacker-i-a-spart-whatsapp-ul-lui-nea-costel-si-trimite-urari-de-craciun-de-pe-el/",
    "https://www.timesnewroman.ro/monden/tigara-electronica-un-pericol-real-barbat-lovit-cu-sania-dupa-ce-mosul-l-a-confundat-cu-un-horn/",
    "https://www.timesnewroman.ro/monden/un-roman-a-incercat-sa-ridice-stergatoarele-unei-dacii-spring-si-a-ridicat-masina-cu-totul/",
    "https://www.timesnewroman.ro/monden/studiu-cu-mancarea-gatita-de-craciun-romanii-ar-putea-rezista-unui-asediu-de-280-de-zile/",
    "https://www.timesnewroman.ro/monden/romanii-se-roaga-pentru-ninsori-masive-ca-sa-nu-le-mai-poata-ajunge-rudele-in-vizita-de-craciun/",
    "https://www.timesnewroman.ro/monden/nea-costel-votat-cel-mai-frumos-betiv-din-lume-dupa-ce-s-a-facut-muci-la-targul-din-craiova/",
    "https://www.timesnewroman.ro/monden/uita-de-coarnele-de-ren-un-roman-si-a-ornat-masina-cu-maioneza-gogosari-si-castraveti/",
    "https://www.timesnewroman.ro/monden/dani-mocanu-a-venit-in-romania-dar-sustine-ca-e-wiz-khalifa-si-are-de-facut-doar-9-luni/",
    "https://www.timesnewroman.ro/monden/fiindca-taierea-porcului-i-se-pare-un-obicei-barbar-un-roman-il-mananca-intreg/",
    "https://www.timesnewroman.ro/monden/youtuberul-care-l-a-batut-pe-andrew-tate-spune-ca-a-fost-un-meci-sub-medie/",
    "https://www.timesnewroman.ro/monden/continua-retragerile-trupelor-americane-din-europa-trupa-metallica-si-a-anulat-concertele/",
    "https://www.timesnewroman.ro/monden/studiu-60-din-barbatii-romani-baga-cu-mana-bila-in-kendama/",
    "https://www.timesnewroman.ro/monden/cruzime-fara-margini-un-roman-sadic-a-asomat-porcul-folosind-muzica-lui-smiley/",
    "https://www.timesnewroman.ro/monden/cea-mai-tanara-fana-a-lui-banica-jr-micuta-de-doar-63-de-ani-stie-toate-cantecele-artistului/",
    "https://www.timesnewroman.ro/monden/un-roman-nu-poate-folosi-casele-self-scan-din-cauza-ca-are-tatuaj-cu-cod-de-bare-pe-mana/",
    "https://www.timesnewroman.ro/monden/singur-acasa-e-nimic-in-ultima-vacanta-cristi-borcea-a-uitat-acasa-6-copii/",
    "https://www.timesnewroman.ro/monden/mobilizare-la-vidraru-catalin-botezatu-s-a-dus-cu-un-grup-de-voluntari-sa-impinga-namolul-la-deal/",
    "https://www.timesnewroman.ro/monden/baraj-din-vaslui-golit-in-2-ore-dupa-ce-un-glumet-a-anuntat-ca-pe-fund-e-o-sticla-de-bautura/",
    "https://www.timesnewroman.ro/monden/dani-mocanu-apel-disperat-salvati-pestii-din-prahova-au-ramas-pe-uscat/",
    "https://www.timesnewroman.ro/monden/milioane-de-romani-abia-asteapta-antena-satelor-wrapped-ca-sa-afle-cat-dolanescu-au-ascultat-in-2025/",
    "https://www.timesnewroman.ro/life-death/mosul-e-sarac-anul-asta-romanii-au-gasit-sub-brad-doar-pliante-de-la-kaufland/",
    "https://www.timesnewroman.ro/life-death/zalaul-e-si-anul-acesta-cel-mai-cautat-oras-din-romania-nimeni-nu-stie-unde-e/",
    "https://www.timesnewroman.ro/life-death/7-indicii-ca-vecinul-tau-clujean-are-lepra/",
    "https://www.timesnewroman.ro/life-death/romantic-de-craciun-un-roman-i-a-oferit-cel-mai-frumos-cadou-sotiei-divortul/",
    "https://www.timesnewroman.ro/life-death/istorie-cine-a-fost-groful-soros-care-a-uneltit-la-asasinarea-lui-mihai-viteazul/",
    "https://www.timesnewroman.ro/life-death/oferta-cei-care-se-duc-in-armata-pot-zbura-de-pe-aeroportul-militar-nu-mai-stau-la-cozi-la-otopeni/",
    "https://www.timesnewroman.ro/life-death/sunt-si-vesti-bune-anul-asta-e-primul-in-care-romanii-nu-se-vor-ingrasa-de-sarbatori/",
    "https://www.timesnewroman.ro/life-death/schimbari-in-codul-fiscal-daca-visezi-ca-iei-bani-dupa-trezire-trebuie-sa-platesti-impozit-pe-ei/",
    "https://www.timesnewroman.ro/life-death/ultimul-prost-care-mai-platea-impozite-in-romania-rasufla-usurat-anul-asta-n-a-avut-venituri/",
    "https://www.timesnewroman.ro/life-death/ospitalitate-romaneasca-lepra-din-cluj-a-fost-batuta-intr-un-gang-de-bacteriile-din-spitale/",
    "https://www.timesnewroman.ro/life-death/clujenii-sunt-panicati-ca-dupa-lepra-s-ar-putea-intoarce-in-oras-si-dominatia-maghiara/",
    "https://www.timesnewroman.ro/life-death/un-roman-sustine-ca-de-cand-are-noul-buletin-cu-cip-i-s-a-inversat-procesul-de-chelire/",
    "https://www.timesnewroman.ro/life-death/doi-romani-care-au-alergat-dupa-un-tren-cfr-au-ajuns-inaintea-lui-la-destinatie/",
    "https://www.timesnewroman.ro/life-death/tramvaiul-5-revine-din-decembrie-mii-de-corporatisti-abia-asteapta-sa-mearga-acasa-dupa-8-ani/",
    "https://www.timesnewroman.ro/life-death/vidul-din-capul-romanilor-care-au-alergat-pe-pista-la-un-pas-sa-aspire-avionul-cu-totul/",
    "https://www.timesnewroman.ro/life-death/studiu-femeile-considera-ca-singurul-spatiu-personal-al-barbatului-este-cosciugul/",
    "https://www.timesnewroman.ro/life-death/sfat-din-batrani-pentru-o-viata-sanatoasa-vinul-rosu-trebuie-baut-dupa-o-tuica/",
    "https://www.timesnewroman.ro/life-death/targul-de-craciun-din-mizil-anulat-dupa-ce-s-a-ars-si-ultimul-beculet-din-instalatie/",
    "https://www.timesnewroman.ro/life-death/breaking-nasa-declara-targul-de-craciun-din-craiova-cel-mai-frumos-din-univers/",
    "https://www.timesnewroman.ro/life-death/olguta-a-intrebat-la-nasa-cat-ar-costa-sa-declare-targul-ei-de-craciun-cel-mai-frumos-din-univers/",
    "https://www.timesnewroman.ro/life-death/exces-de-zel-un-paznic-de-farmacie-e-atat-de-capabil-ca-nu-intra-nimeni/",
    "https://www.timesnewroman.ro/life-death/preturi-nesimtite-e-mai-ieftin-sa-te-duci-o-data-pe-luna-cu-low-costul-in-elvetia-la-cumparaturi/",
    "https://www.timesnewroman.ro/life-death/sfatul-nutritionistilor-branza-in-coaja-de-brad-trebuie-mancata-cu-tot-cu-coaja/",
    "https://www.timesnewroman.ro/life-death/carrefour-are-un-nou-proprietar-cine-e-romanul-care-l-a-luat-gratis-dupa-ce-a-uitat-sa-l-scaneze-la-casa/",
    "https://www.timesnewroman.ro/life-death/mapn-a-trimis-100-000-de-invitatii-la-nunta-ca-sa-scape-toti-cei-100-000-s-au-inrolat-in-armata/",
    "https://www.timesnewroman.ro/life-death/un-inconstient-s-a-mutat-in-militari-residence-desi-nu-stie-sa-inoate/",
    "https://www.timesnewroman.ro/life-death/grija-mare-la-razboiul-hibrid-spionii-rusi-i-au-inchis-telefonul-si-l-au-tinut-trei-zile-intr-o-carciuma/",
    "https://www.timesnewroman.ro/life-death/val-de-scumpiri-doar-primele-2-roalert-uri-vor-fi-gratuite-urmatoarele-vor-costa-2-euro-bucata/",
    "https://www.timesnewroman.ro/life-death/o-scoala-a-angajat-din-greseala-un-folkist-in-loc-de-fochist-copiii-stau-in-frig-dar-au-invatat-rapa/",
    "https://www.timesnewroman.ro/life-death/un-roman-a-facut-atatea-accidente-ca-si-a-deschis-firma-de-evenimente-rutiere/"
]


def scrape_and_save(url, label, index):
    """
    Descarcă articolul, salvează imaginea și returnează datele pentru CSV.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Extragere Titlu (Adaptabil în funcție de site)
        title_tag = soup.find('h1')
        if not title_tag:
            return None
        title = title_tag.get_text().strip()

        # 2. Extragere Text (Paragrafe)
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text() for p in paragraphs])

        # Curățare text (eliminăm spații duble și text prea scurt)
        text_content = " ".join(text_content.split())
        if len(text_content) < 100:
            return None

        # Combinăm titlul cu textul pentru context mai bun
        full_text = f"{title}. {text_content[:2000]}"  # Limităm la 2000 caractere

        # 3. Extragere Imagine
        image_url = ""
        meta_image = soup.find('meta', property='og:image')
        if meta_image:
            image_url = meta_image['content']
        else:
            # Fallback
            img = soup.find('img')
            if img and 'src' in img.attrs:
                image_url = img['src']

        if not image_url or not image_url.startswith('http'):
            return None

        # Descărcare imagine
        img_resp = requests.get(image_url, headers=headers, timeout=5)
        img = Image.open(BytesIO(img_resp.content)).convert('RGB')

        # Salvare Imagine
        filename = f"{label}_{index}.jpg"
        folder_name = "real" if label == 0 else "fake"
        save_path = os.path.join(BASE_DIR, folder_name, filename)
        img.save(save_path)

        print(f"✅ [SUCCESS] {folder_name.upper()}: {title[:30]}...")

        return {
            "text": full_text,
            "filename": filename,
            "folder": folder_name,
            "label": label  # 0 = Real, 1 = Fake
        }

    except Exception as e:
        print(f"❌ [ERROR] {url}: {e}")
        return None


def main():
    dataset_rows = []

    print("--- Începe descărcarea datelor REALE ---")
    for i, url in enumerate(urls_real):
        data = scrape_and_save(url, label=0, index=i)
        if data:
            dataset_rows.append(data)
        time.sleep(1)  # Pauză să nu blocăm serverul

    print("\n--- Începe descărcarea datelor FAKE ---")
    for i, url in enumerate(urls_fake):
        data = scrape_and_save(url, label=1, index=i)
        if data:
            dataset_rows.append(data)
        time.sleep(1)

    # Salvare CSV
    df = pd.DataFrame(dataset_rows)
    csv_path = os.path.join(BASE_DIR, "dataset_index.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n🎉 Gata! Dataset creat cu {len(df)} articole.")
    print(f"Index salvat în: {csv_path}")


if __name__ == "__main__":
    main()