"""Doplnenie textu práce do šablóny .docx.

Skript prepíše telo dokumentu (kapitoly 1–6, záver, literatúru a prílohy)
skutočným textom a číslami z `reports/tables` a `reports/figures`.

Použitie (z priečinka `aplikacia`):
    python tools/build_praca.py

Po spustení odporúčame v PowerPointe/Worde aktualizovať polia (Ctrl+A, F9),
aby sa doplnil obsah a čísla strán.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx_utils import (  # noqa: E402
    DocumentBuilder,
    clear_between,
    enable_field_updates,
)

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DOCX_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "Praca_SOC_Predikcia_vysledkov_futbalovych_zapasov.docx"
)
TABLES = PROJECT / "reports" / "tables"
FIGURES = PROJECT / "reports" / "figures"
METADATA = json.loads((PROJECT / "models" / "metadata.json").read_text(encoding="utf-8"))

TITLE = "Predikcia výsledkov futbalových zápasov"


# ---------------------------------------------------------------------------
# Formátovanie čísel v slovenskej konvencii
# ---------------------------------------------------------------------------
def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def signed_pct(value: float, digits: int = 1) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.{digits}f} %".replace(".", ",")


def load_tables() -> dict[str, pd.DataFrame]:
    name_map = {
        "comparison": "04_model_comparison.csv",
        "per_class": "05_per_class_metrics.csv",
        "confusion": "06_confusion_matrices.csv",
        "odds": "07_odds_comparison.csv",
        "value": "08_value_bets.csv",
        "confidence": "09_accuracy_by_confidence.csv",
    }
    return {
        key: pd.read_csv(TABLES / filename, encoding="utf-8")
        for key, filename in name_map.items()
    }


# ---------------------------------------------------------------------------
# Úpravy prednej časti dokumentu
# ---------------------------------------------------------------------------
def strip_heading_numbering(document) -> None:
    """Odstráni automatické číslovanie z nadpisov.

    Čísla kapitol píšeme priamo v texte (napr. „3 MATERIÁL A METODIKA“),
    takže automatické číslovanie štýlov by ich v obsahu zdvojilo.
    """
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        ppr = document.styles[name].element.find(qn("w:pPr"))
        if ppr is None:
            continue
        num_pr = ppr.find(qn("w:numPr"))
        if num_pr is not None:
            ppr.remove(num_pr)


def _single_right_tab(paragraph, stop_cm: float = 15.0) -> None:
    """Nahradí sériu tabulátorov jedným zarovnaním k pravému okraju.

    Šablóna používa sedem po sebe idúcich tabulátorov; pri dlhšom
    placeholderi sa potom riadok láme. Jeden pravý tabulátor je
    stabilnejší a výsledok vyzerá rovnako.
    """
    first = next((run for run in paragraph.runs if "\t" in run.text), None)
    if first is None:
        return
    for run in paragraph.runs:
        if "\t" in run.text:
            run.text = run.text.replace("\t", "")
    tab_run = OxmlElement("w:r")
    tab_run.append(OxmlElement("w:tab"))
    first._r.addprevious(tab_run)
    paragraph.paragraph_format.tab_stops.add_tab_stop(
        Cm(stop_cm), WD_TAB_ALIGNMENT.RIGHT
    )


def fix_front_matter(document) -> None:
    import re

    paragraphs = document.paragraphs

    # Rok na obálke a titulnom liste (v šablóne je rozdelený do dvoch behov).
    for paragraph in paragraphs[:60]:
        runs = paragraph.runs
        for index in range(len(runs) - 1):
            if runs[index].text + runs[index + 1].text == "2025":
                runs[index].text = "2026"
                runs[index + 1].text = ""
        for run in runs:
            if run.text.strip() == "2025":
                run.text = run.text.replace("2025", "2026")
        if paragraph.text.startswith("Konzultant"):
            for run in runs:
                if run.text.strip() == "xxx":
                    run.text = ""
            # Riadok je dlhý; bez zmenšenia odsadenia sa láme do dvoch riadkov.
            fmt = paragraph.paragraph_format
            fmt.left_indent = Cm(0)
            fmt.first_line_indent = Cm(0)
            fmt.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # Téma práce namiesto XXX v čestnom vyhlásení; v poďakovaní zostáva
    # placeholder na meno konzultanta.
    for paragraph in paragraphs:
        if "XXX" not in paragraph.text:
            continue
        replacement = (
            TITLE
            if "Vyhlasujem" in paragraph.text
            else "[Doplniť: Titul, Meno a Priezvisko]"
        )
        for run in paragraph.runs:
            if "XXX" in run.text:
                run.text = run.text.replace("XXX", replacement)

    # Na druhej titulnej strane je riešiteľ prilepený k číslu odboru.
    for paragraph in paragraphs:
        if paragraph.text.startswith("Č. odboru") and "[Doplniť: Meno" in paragraph.text:
            for run in paragraph.runs:
                if run.text.startswith("[Doplniť: Meno"):
                    run.text = "\t" + run.text

    # Riadky s riešiteľom a ročníkom zarovnáme jedným pravým tabulátorom.
    for paragraph in paragraphs[:45]:
        text = paragraph.text.strip()
        if (
            "\t" in paragraph.text
            and "dňa" not in text
            and text.startswith(("Stará Ľubovňa", "2026", "Č. odboru"))
        ):
            _single_right_tab(paragraph)

    # Poznámky šablóny "(nepovinná časť)" a pokyn k abecednému poradiu
    # v hotovej práci nemajú čo robiť.
    for paragraph in paragraphs:
        if "nepovinná časť" in paragraph.text:
            for run in paragraph.runs:
                if "nepovinná časť" in run.text:
                    run.text = re.sub(r"\s*\(?nepovinná časť\)?", "", run.text).rstrip()
        if paragraph.text.strip().startswith("*uvádzať"):
            paragraph._p.getparent().remove(paragraph._p)


def fix_front_pagination(document) -> None:
    """Zmenší prázdny odsek na konci poľa obsahu.

    Políčko obsahu končí prázdnym odsekom, ktorý obsahuje ukončenie poľa
    (`fldChar`). Ak sa nezmestí na stranu s obsahom, vznikne pred zoznamom
    skratiek prázdna strana. Zmenšením riadku na 1 pt sa odsek vmestí.
    """
    body = document.element.body
    sdt = next((child for child in body if child.tag == qn("w:sdt")), None)
    if sdt is None:
        return
    content = sdt.find(qn("w:sdtContent"))
    if content is None or not len(content):
        return

    last = content[-1]
    if last.tag != qn("w:p") or "".join(last.itertext()).strip():
        return

    ppr = last.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        last.insert(0, ppr)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "0")
    spacing.set(qn("w:line"), "20")
    spacing.set(qn("w:lineRule"), "exact")
    ppr.append(spacing)


def rebuild_abbreviations(document) -> None:
    """Vyplní zoznam skratiek a odstráni prázdne odrážky šablóny."""
    items = [
        ("B365", "kurzy stávkovej kancelárie Bet365"),
        ("CI", "interval spoľahlivosti (confidence interval)"),
        ("CSV", "textový formát s hodnotami oddelenými čiarkou"),
        ("Elo", "ratingová metóda na odhad relatívnej sily tímov"),
        ("F1", "harmonický priemer presnosti a úplnosti klasifikácie"),
        ("GB", "gradient boosting (postupné stromové modely)"),
        ("H2H", "vzájomné zápasy dvoch tímov (head-to-head)"),
        ("ML", "strojové učenie (machine learning)"),
        ("p.p.", "percentuálny bod"),
        ("RF", "náhodný les (random forest)"),
        ("ROI", "návratnosť vkladu (return on investment)"),
        ("SOČ", "stredoškolská odborná činnosť"),
        ("xG", "očakávané góly (expected goals)"),
    ]

    texts = [f"**{short}** – {meaning}" for short, meaning in items]
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.startswith("Zoznam skratiek"))
    placeholder = [
        p for p in paragraphs[start + 1 : start + 40] if p.style.name == "List Paragraph"
    ]
    if not placeholder:
        return

    builder = DocumentBuilder(document, placeholder[0])
    for text in texts:
        builder.paragraph(text, indent=False, space_after=Pt(2))

    for paragraph in placeholder:
        paragraph._p.getparent().remove(paragraph._p)


def rebuild_lists(document) -> None:
    """Doplní zoznam tabuliek, grafov a ilustrácií."""
    entries = [
        "Tab. 1: Rozdelenie dátových množín",
        "Tab. 2: Prehľad vybraných predzápasových atribútov",
        "Tab. 3: Zvolené hyperparametre modelov",
        "Tab. 4: Porovnanie modelov na testovacej sezóne",
        "Tab. 5: Metriky po jednotlivých triedach výsledku",
        "Tab. 6: Presnosť modelov podľa pásma istoty",
        "Tab. 7: Porovnanie modelov s kurzami stávkovej kancelárie",
        "Tab. 8: Simulácia value bets a referenčné stratégie",
        "Graf 1: Rozdelenie výsledkov zápasov",
        "Graf 2: Rozdelenie počtu gólov v zápase",
        "Graf 3: Dôležitosť atribútov v náhodnom lese",
        "Graf 4: Matice zámen modelov a kurzového trhu",
        "Graf 5: Kalibrácia pravdepodobností",
        "Graf 6: Návratnosť simulovaných value bets",
        "Obr. 1: Predikcia zápasu Arsenal – Chelsea v aplikácii",
        "Obr. 2: Porovnanie formy tímov v aplikácii",
    ]

    paragraphs = document.paragraphs
    heading_index = next(
        i for i, p in enumerate(paragraphs) if p.text.startswith("Zoznam tabuliek")
    )
    # Pôvodné ukážkové riadky (Tab. 1 / Graf 1) odstránime.
    for paragraph in [
        p
        for p in paragraphs[heading_index : heading_index + 10]
        if p.text.startswith(("Tab. 1:", "Graf 1:", "*napr."))
    ]:
        paragraph._p.getparent().remove(paragraph._p)

    anchor = next(
        p
        for p in document.paragraphs[heading_index:]
        if p.text.strip() == "" and "sectPr" in p._p.xml
    )
    builder = DocumentBuilder(document, anchor)
    for entry in entries:
        builder.paragraph(entry, indent=False, space_after=Pt(2))


def insert_toc(document) -> None:
    """Obsah je v šablóne už vložený ako pole (sdt) – netreba druhý.

    Šablóna obsahuje pole TOC, ktoré Word aktualizuje pri otvorení
    dokumentu (`w:updateFields`). Vkladanie ďalšieho poľa by vytvorilo
    dva obsahy za sebou.
    """
    return


# ---------------------------------------------------------------------------
# Kapitoly
# ---------------------------------------------------------------------------
def build_chapters(builder: DocumentBuilder) -> None:
    from docx.shared import Pt

    # ------------------------------------------------------------------
    # ÚVOD
    # ------------------------------------------------------------------
    builder.heading("ÚVOD")
    builder.paragraph(
        "Futbal je jedným z najpopulárnejších športov na svete a anglická Premier "
        "League patrí medzi najsledovanejšie ligové súťaže. Každý zápas prináša "
        "veľké množstvo číselných údajov – góly, strely, držanie lopty, kurzy "
        "stávkových kancelárií – ktoré sa dajú spracovať metódami dátovej analýzy. "
        "Práve preto je futbal vhodnou oblasťou na overenie toho, čo dokáže "
        "strojové učenie: či sa z historických výsledkov dá odvodiť model, ktorý "
        "odhadne výsledok budúceho zápasu lepšie než náhodný tip alebo než "
        "jednoduché pravidlo „vyhrá domáci tím“."
    )
    builder.paragraph(
        "Výsledok zápasu je zároveň ukážkou problému, ktorý je v praxi veľmi "
        "typický. Máme málo dát (v jednej sezóne iba 380 zápasov), značný podiel "
        "náhody a tri možnosti, ktoré nie sú rovnako časté. Model preto nemôže "
        "byť postavený na jednom ukazovateli, ale musí kombinovať viacero "
        "predzápasových informácií: formu tímu, kvalitu útoku a obrany, silu "
        "súpera, domáce prostredie či vzájomnú bilanciu. Práca ukazuje celý "
        "postup od stiahnutia dát až po webovú aplikáciu, ktorá predikciu "
        "zobrazí používateľovi."
    )
    builder.paragraph(
        "Práca je rozdelená do šiestich kapitol. Prvá kapitola vymedzuje "
        "teoretický základ strojového učenia, špecifiká predikcie športových "
        "výsledkov a úlohu kurzov stávkových kancelárií. Druhá kapitola "
        "formuluje hlavný cieľ a čiastkové ciele. Tretia kapitola opisuje dáta, "
        "tvorbu atribútov, použité modely a spôsob vyhodnotenia. Štvrtá kapitola "
        "prináša konkrétne výsledky vrátane porovnania s kurzami a ukážok "
        "aplikácie. Piata kapitola výsledky interpretuje a pomenúva obmedzenia, "
        "šiesta kapitola prácu zhrňuje."
    )

    # ------------------------------------------------------------------
    # 1 TEORETICKÁ ČASŤ
    # ------------------------------------------------------------------
    builder.page_break()
    builder.heading("1 STROJOVÉ UČENIE A DÁTOVÁ ANALYTIKA VO FUTBALE")
    builder.paragraph(
        "Táto kapitola vymedzuje základný teoretický aparát potrebný na "
        "pochopenie problematiky: čo je strojové učenie, ako funguje "
        "klasifikácia, aké sú špecifiká predikcie futbalových zápasov a akú "
        "úlohu pri tom zohrávajú kurzy stávkových kancelárií."
    )

    builder.heading("1.1 Strojové učenie a klasifikácia", 2)
    builder.paragraph(
        "Strojové učenie (machine learning) je odvetvie umelej inteligencie, "
        "ktoré skúma algoritmy schopné zlepšovať svoje výsledky na základe "
        "skúseností zapísaných v dátach. Namiesto ručne napísaných pravidiel "
        "dostane algoritmus množinu príkladov a hľadá v nich vzory, ktoré potom "
        "dokáže uplatniť na nové, dovtedy nevidené prípady. Ak sú príklady "
        "označené správnym výsledkom, hovoríme o učení s učiteľom "
        "(supervised learning)."
    )
    builder.paragraph(
        "V našej úlohe je vstupom **vektor atribútov** (napríklad forma oboch "
        "tímov a ich Elo rating) a výstupom **trieda** – výhra domácich (H), "
        "remíza (D) alebo výhra hostí (A). Ide teda o **multitriednu "
        "klasifikáciu**. Kvalitný model nevracia len jednu triedu, ale "
        "pravdepodobnostné rozdelenie pre všetky tri možnosti. To je dôležité "
        "preto, lebo z pravdepodobností sa dá počítať log loss, kalibrácia aj "
        "prípadná výhoda oproti kurzom. Predikovaná trieda je potom tá "
        "s najvyššou pravdepodobnosťou."
    )
    builder.paragraph(
        "Dáta sa pred trénovaním rozdeľujú na tri časti. **Trénovacia množina** "
        "slúži na učenie parametrov, **validačná množina** na výber "
        "hyperparametrov a porovnanie modelov a **testovacia množina** na "
        "záverečné, nezaujaté vyhodnotenie. Ak by sa testovacia množina "
        "používala pri ladení, výsledky by boli nadhodnotené – model by sa "
        "učil „na test“. Preto sa v tejto práci testovacia sezóna použila iba "
        "raz, až po uzavretí všetkých rozhodnutí."
    )
    builder.paragraph(
        "V práci používame tri algoritmy. **Logistická regresia** je lineárny "
        "model, ktorý odhaduje pravdepodobnosti jednotlivých tried pomocou "
        "logistickej funkcie; je jednoduchá, stabilná a dobre "
        "interpretovateľná, ale zachytí len lineárne vzťahy. **Náhodný les** "
        "(random forest) kombinuje mnoho rozhodovacích stromov trénovaných na "
        "náhodných podmnožinách dát a atribútov; zvláda nelineárne vzťahy a "
        "býva odolný voči preučeniu. **Gradient boosting** stavia stromy "
        "postupne tak, že každý ďalší strom opravuje chyby predchádzajúcich; "
        "na tabuľkových dátach patrí k najúčinnejším metódam, no pri malej "
        "vzorke môže byť nestabilný."
    )
    builder.paragraph(
        "Kvalitu modelu hodnotíme viacerými metrikami, pretože každá zachytáva "
        "inú stránku. **Presnosť (accuracy)** udáva podiel správne "
        "predikovaných zápasov, no pri nevyvážených triedach môže klamať – "
        "model, ktorý vždy tipne výhru domácich, má na našich dátach presnosť "
        "okolo 43 %, hoci remízy a výhry hostí netipne nikdy. **Vyvážená "
        "presnosť** počíta priemer úspešnosti jednotlivých tried a **macro "
        "F1** spriemeruje F1-skóre tried, takže slabý výsledok pri remízach "
        "sa v ňom prejaví. **Log loss** trestá príliš sebavedomé nesprávne "
        "pravdepodobnosti a je preto vhodný na porovnanie s kurzami. "
        "**Brierovo skóre** je priemerná kvadratická odchýlka od skutočného "
        "výsledku a citlivejšie reaguje na veľké chyby."
    )

    builder.heading("1.2 Špecifiká predikcie futbalového zápasu", 2)
    builder.paragraph(
        "Futbalový zápas ovplyvňuje niekoľko faktorov, ktoré sa v dátach dajú "
        "zachytiť len približne. Najvýraznejší je **domáce prostredie**: "
        "v skúmaných sezónach skončilo výhrou domácich 44,2 % zápasov, "
        "remízou 23,9 % a výhrou hostí 31,9 %. Tento rozdiel súvisí s "
        "cestovaním, povzbudzovaním publika aj so štýlom hry, ktorý si tímy "
        "volia doma a vonku. Model ho nemusí „pozerať“ priamo – stačí, keď "
        "pracuje s oddelenými štatistikami pre domáce a vonkajšie zápasy."
    )
    builder.paragraph(
        "Druhým špecifikom je **nízka skórovanosť**. Zápas má v priemere "
        "necelé tri góly, takže jediná náhodná udalosť – červená karta, "
        "nešťastný odraz, penalta – môže zvrátiť výsledok. Práve preto je "
        "strop presnosti pri futbalových predikciách pomerne nízky: "
        "aj skúsení analytici a stávkové kancelárie sa pohybujú okolo 50–55 %."
    )
    builder.paragraph(
        "Tretím špecifikom je **nerovnomerné zastúpenie tried**. Výhry "
        "domácich sú najčastejšie, remízy najmenej časté a zároveň "
        "najťažšie predikovateľné, pretože vznikajú z tesných výsledkov. "
        "Model, ktorý chce mať vysokú presnosť, má preto tendenciu remízy "
        "ignorovať. V práci tento jav nepotláčame, ale meriame ho a "
        "komentujeme – je to legitímne zistenie o povahe úlohy."
    )
    builder.paragraph(
        "Z dostupných dát vieme zostaviť **predzápasové atribúty**: formu za "
        "posledných päť zápasov, priemerný počet strelených a inkasovaných "
        "gólov, strely na bránu (približná náhrada očakávaných gólov), dni "
        "od posledného zápasu, body na zápas v aktuálnej sezóne, vzájomnú "
        "bilanciu a Elo rating. Elo rating je číselné ohodnotenie sily tímu, "
        "ktoré sa po každom zápase aktualizuje podľa toho, ako prekvapivý bol "
        "výsledok; tím, ktorý porazí silnejšieho súpera, získa viac bodov než "
        "tím, ktorý porazí slabšieho."
    )

    builder.heading("1.3 Kurzy stávkových kancelárií a efektívnosť trhu", 2)
    builder.paragraph(
        "Kurzy stávkových kancelárií sú užitočnou referenciou, pretože do nich "
        "vstupuje veľké množstvo informácií – nielen forma a zranenia, ale aj "
        "očakávania trhu. Z kurzu sa dá vypočítať **implikovaná "
        "pravdepodobnosť** ako prevrátená hodnota kurzu (1 / kurz). Súčet "
        "implikovaných pravdepodobností je väčší ako 1, pretože kancelária si "
        "drží maržu; po normalizácii (vydelení súčtom) dostaneme odhad "
        "pravdepodobností, ktorý už maržu neobsahuje."
    )
    builder.paragraph(
        "Ak je trh efektívny, jeho pravdepodobnosti by mali byť veľmi blízko "
        "skutočným frekvenciám a mali by mať nízky log loss. Pre model to "
        "znamená tvrdú konkurenciu: nestačí prekonať náhodný tip, treba sa "
        "priblížiť k trhu. V práci preto kurzy **nevstupujú do modelu** – "
        "inak by model len kopíroval trh a neukázal by, čo sa dá zistiť "
        "z výsledkov a štatistík. Slúžia výhradne na porovnanie."
    )
    builder.paragraph(
        "Z kurzov sa dá simulovať aj stávkovanie. Ak model odhadne "
        "pravdepodobnosť triedy vyššiu, než akú pripisuje trh, hovoríme o "
        "**value bet** – stávke, ktorá má podľa modelu kladnú očakávanú "
        "hodnotu. Výnos takejto stratégie meriame pomocou **ROI** (návratnosť "
        "vkladu): ak na každú stávku vsadíme jednu jednotku, ROI je priemerný "
        "zisk na jednu stávku. Keďže ide o jedinú sezónu, dopĺňame ROI "
        "bootstrapovým intervalom spoľahlivosti, ktorý ukazuje, aká veľká "
        "neistota je v odhade."
    )

    # ------------------------------------------------------------------
    # 2 CIELE
    # ------------------------------------------------------------------
    builder.page_break()
    builder.heading("2 CIELE PRÁCE")
    builder.paragraph(
        "Hlavným cieľom práce je navrhnúť, implementovať a vyhodnotiť "
        "predikčný systém, ktorý pomocou algoritmov strojového učenia "
        "v jazyku Python odhaduje výsledky futbalových stretnutí anglickej "
        "Premier League, a vytvoriť funkčnú webovú aplikáciu, ktorá predikciu "
        "zobrazí používateľovi."
    )
    builder.paragraph("Čiastkové ciele práce:", space_after=Pt(3))
    goals = [
        "Získať a vyčistiť historické dáta o zápasoch Premier League vrátane kurzov.",
        "Vytvoriť predzápasové indikátory formy a sily tímov tak, aby žiadny atribút neobsahoval informáciu z budúcnosti.",
        "Natrénovať a navzájom porovnať viacero klasifikačných modelov a vybrať model podľa validačnej sezóny.",
        "Zhodnotiť úspešnosť predikcií voči kurzom stávkových kancelárií a simulovať stávkovanie na základe modelu.",
        "Vytvoriť webové rozhranie v knižnici Streamlit, ktoré umožní predikciu ľubovoľnej dvojice tímov.",
    ]
    for index, goal in enumerate(goals, start=1):
        builder.paragraph(f"{index}. {goal}", indent=True, space_after=Pt(3))

    # ------------------------------------------------------------------
    # 3 MATERIÁL A METODIKA
    # ------------------------------------------------------------------
    builder.page_break()
    builder.heading("3 MATERIÁL A METODIKA")

    builder.heading("3.1 Dátová základňa", 2)
    builder.paragraph(
        "Pre účely práce boli použité historické dáta vo formáte CSV z portálu "
        "football-data.co.uk, ktorý zverejňuje výsledky a kurzy európskych "
        "ligových súťaží. Spracovali sme päť kompletných sezón anglickej "
        "Premier League: 2021/22 až 2025/26. Každá sezóna obsahuje 380 zápasov "
        "a 20 tímov, spolu teda **1 900 zápasov**."
    )
    builder.paragraph(
        "Dataset zahŕňa výsledky zápasov, góly v polčase, strely, strely na "
        "bránu, rohy, fauly, žlté a červené karty a predzápasové kurzy "
        "stávkových kancelárií Bet365 a Pinnacle. Kurzy sme do modelu "
        "nezaradili, použili sme ich len na porovnanie. Základné rozdelenie "
        "dát je v tabuľke 1."
    )

    comparison = load_tables()["comparison"]
    builder.table(
        "Tab. 1: Rozdelenie dátových množín",
        ["Množina", "Sezóny", "Počet zápasov", "Podiel"],
        [
            ["Trénovacia", "2021/22 – 2023/24", "1 140", "60,0 %"],
            ["Validačná", "2024/25", "380", "20,0 %"],
            ["Testovacia", "2025/26", "380", "20,0 %"],
            ["Spolu", "2021/22 – 2025/26", "1 900", "100,0 %"],
        ],
        widths=[3.5, 5.0, 3.5, 3.0],
    )
    builder.paragraph(
        "Rozdelenie je striktne chronologické: model sa učí na starších "
        "sezónach a testuje na najnovšej. Takéto rozdelenie zodpovedá reálnemu "
        "použitiu, kde predikujeme budúcnosť z minulosti. Náhodné rozdelenie "
        "zápasov by bolo nesprávne, pretože by model videl zápasy z tej istej "
        "sezóny, ktoré časovo nasledujú po testovaných."
    )

    builder.heading("3.2 Tvorba predzápasových atribútov", 2)
    builder.paragraph(
        "Zápasy sme previedli do takzvaného dlhého formátu, v ktorom má každý "
        "zápas dva riadky – jeden pre domáci a jeden pre hosťujúci tím. "
        "Riadky sú zoradené chronologicky podľa tímu a všetky kĺzavé "
        "štatistiky sa počítajú s posunom o jeden zápas (`shift(1)`). "
        "Atribút zápasu teda nikdy neobsahuje výsledok tohto zápasu, iba "
        "predchádzajúce zápasy. Rovnaké pravidlo platí pre Elo rating a pre "
        "vzájomnú bilanciu: aktualizujú sa až po vyhodnotení zápasu."
    )
    builder.paragraph(
        "Celkovo vzniklo **46 atribútov**. Forma sa počíta z posledných piatich "
        "zápasov (body, výhry, remízy, strelené a inkasované góly, strely na "
        "bránu, rohy), oddelene pre domáce a vonkajšie zápasy. K tomu "
        "pristupuje počet dní od posledného zápasu, body na zápas v aktuálnej "
        "sezóne, Elo rating, príznak postupujúceho tímu, vzájomná bilancia a "
        "odvodené rozdiely (napríklad rozdiel Elo ratingov alebo rozdiel formy). "
        "Vybrané atribúty sú v tabuľke 2."
    )
    builder.table(
        "Tab. 2: Prehľad vybraných predzápasových atribútov",
        ["Atribút", "Význam", "Okno"],
        [
            ["form_pts_5", "body z posledných zápasov (V = 3, R = 1, P = 0)", "5"],
            ["form_pts_venue_5", "body len z domácich, resp. vonkajších zápasov", "5"],
            ["gf_avg_5, ga_avg_5", "priemer strelených a inkasovaných gólov", "5"],
            ["sot_for_avg_5", "priemer striel na bránu (náhrada xG)", "5"],
            ["shots_avg_5, corners_avg_5", "priemer striel a rohov", "5"],
            ["rest_days", "dni od posledného zápasu (max. 30)", "–"],
            ["season_pts_pg", "body na zápas v aktuálnej sezóne", "sezóna"],
            ["elo", "Elo rating pred zápasom (štart 1500, K = 20)", "–"],
            ["h2h_pts_H, h2h_gd_H", "body a gólový rozdiel z posledných vzájomných zápasov", "5"],
            ["is_promoted", "tím nebol v predchádzajúcej sezóne v lige", "–"],
            ["elo_diff, form_diff", "rozdiely atribútov domácich a hostí", "–"],
            ["attack_defense", "útok jedného tímu proti obrane druhého", "5"],
        ],
        widths=[4.4, 8.0, 2.6],
    )
    builder.paragraph(
        "Prvé zápasy tímu v datasete nemajú históriu. Namiesto chýbajúcich "
        "hodnôt sme použili rozumné futbalové konštanty (napríklad 1,35 "
        "streleného gólu na zápas alebo 6,75 bodu z piatich zápasov). Tieto "
        "hodnoty nie sú vypočítané z budúcich dát, takže neporušujú pravidlo "
        "predzápasovosti. Rovnako sme ošetrili postupujúce tímy príznakom "
        "`is_promoted`, ktorý modelu naznačuje, že ich história je krátka."
    )
    builder.paragraph(
        "Správnosť výpočtu overujú automatické testy. Jeden test odstráni "
        "posledný zápas datasetu a skontroluje, že atribúty ostatných zápasov "
        "sa nezmenili. Ďalší zmení výsledok zápasu na opačný a overí, že sa "
        "nezmenili atribúty tohto zápasu ani žiadneho skoršieho zápasu. "
        "Testy sú súčasťou projektu a spúšťajú sa príkazom `pytest tests`."
    )

    builder.heading("3.3 Modely a ladenie hyperparametrov", 2)
    builder.paragraph(
        "Porovnávali sme tri modely: logistickú regresiu s normalizáciou "
        "vstupov, náhodný les so 400 stromami a gradient boosting. "
        "Hyperparametre sme hľadali pomocou `RandomizedSearchCV` s dvanástimi "
        "kombináciami a krížovou validáciou `TimeSeriesSplit` o piatich "
        "častiach, ktorá zachováva chronologické poradie. Kritériom bola "
        "hodnota log loss na trénovacej množine. Po výbere parametrov sa "
        "model dotrénoval na trénovacej a validačnej sezóne a až potom sa "
        "raz vyhodnotil na testovacej sezóne 2025/26."
    )
    builder.paragraph(
        "Najprv sme skúsili aj `HistGradientBoostingClassifier`, ktorý je "
        "rýchlejší a býva odporúčaný pre väčšie datasety. Na našej vzorke "
        "1 140 zápasov však dával validačný log loss 1,17, teda výrazne horší "
        "než ostatné modely, a jeho pravdepodobnosti boli zle kalibrované. "
        "Preto sme ho nahradili klasickým gradient boostingom, ktorý dosiahol "
        "validačný log loss 0,995. Toto rozhodnutie uvádzame aj v diskusii, "
        "pretože ukazuje, že voľba modelu závisí od konkrétnych dát."
    )

    models = METADATA["models"]
    lr = models["logistic_regression"]
    rf = models["random_forest"]
    gb = models["gradient_boosting"]
    builder.table(
        "Tab. 3: Zvolené hyperparametre modelov",
        ["Model", "Zvolené nastavenie", "Log loss (validácia)"],
        [
            [
                "Logistická regresia",
                "C = 0,0083; class_weight = balanced",
                num(lr["valid_metrics"]["log_loss"], 4),
            ],
            [
                "Náhodný les",
                "400 stromov; max_depth = 16; min_samples_leaf = 10; max_features = 0,5",
                num(rf["valid_metrics"]["log_loss"], 4),
            ],
            [
                "Gradient boosting",
                "learning_rate = 0,02; 200 stromov; max_depth = 2; subsample = 0,8",
                num(gb["valid_metrics"]["log_loss"], 4),
            ],
        ],
        widths=[3.6, 8.6, 2.8],
    )
    builder.paragraph(
        "Logistická regresia dostala relatívne silnú regularizáciu (C = 0,0083) "
        "a vyváženie tried, pretože remízy sú menej časté. Náhodný les má "
        "hĺbku 16 a minimálne 10 zápasov v liste, čo bráni preučeniu. "
        "Gradient boosting má plytkú hĺbku 2 a nižšiu rýchlosť učenia 0,02 – "
        "na malej vzorke sa takto správajú stabilnejšie."
    )

    builder.heading("3.4 Metriky a spôsob vyhodnotenia", 2)
    builder.paragraph(
        "Modely hodnotíme na testovacej sezóne 2025/26. Ako referenciu "
        "používame dva jednoduché prístupy – tip na výhru domácich a náhodný "
        "tip s rovnakou pravdepodobnosťou pre všetky triedy – a tiež "
        "**kurzového favorita**, teda triedu s najvyššou normalizovanou "
        "implikovanou pravdepodobnosťou. Kurzový favorit predstavuje odhad "
        "trhu a je najnáročnejšou referenciou."
    )
    builder.paragraph(
        "Počítame presnosť, vyváženú presnosť, macro F1, log loss a Brierovo "
        "skóre. Súčasťou vyhodnotenia sú matice zámen, ktoré ukazujú, ktoré "
        "triedy si model mýli, a kalibračné krivky, ktoré porovnávajú "
        "predikované pravdepodobnosti so skutočnou frekvenciou. Pri "
        "stávkovaní simulujeme prahy výhody 3 %, 5 %, 8 % a 10 % a ku každému "
        "ROI počítame 95 % interval spoľahlivosti pomocou 2 000 "
        "bootstrapových vzoriek."
    )

    # ------------------------------------------------------------------
    # 4 VÝSLEDKY
    # ------------------------------------------------------------------
    builder.page_break()
    builder.heading("4 VÝSLEDKY PRÁCE")

    builder.heading("4.1 Zber a čistenie dát", 2)
    builder.paragraph(
        "Všetkých 1 900 zápasov prešlo kontrolami bez chyby: žiadny neplatný "
        "dátum, žiadny chýbajúci výsledok, žiadna duplicita a v každej sezóne "
        "presne 20 tímov. Kurzy Bet365 boli dostupné pre všetkých 1 900 "
        "zápasov; kurzy Pinnacle chýbali pri 170 zápasoch sezóny 2025/26, "
        "preto sme pri porovnaní použili Bet365."
    )
    builder.paragraph(
        "Rozdelenie výsledkov je vidieť na grafe 1. Výhry domácich tvoria "
        "44,2 %, výhry hostí 31,9 % a remízy 23,9 %. V testovacej sezóne boli "
        "remízy o niečo častejšie (27,4 %), čo je pri 380 zápasoch bežná "
        "medzisezónna variácia. Graf 2 ukazuje rozdelenie počtu gólov; zápas "
        "má v priemere 2,93 gólu a najčastejšie končí dvoma alebo troma gólmi."
    )
    builder.figure(FIGURES / "01_class_distribution.png", "Graf 1: Rozdelenie výsledkov zápasov", 12.5)
    builder.figure(FIGURES / "02_goals_distribution.png", "Graf 2: Rozdelenie počtu gólov v zápase", 13.5)

    builder.heading("4.2 Charakteristika vytvorených atribútov", 2)
    builder.paragraph(
        "Atribúty majú očakávané hodnoty. Priemerná forma domácich je 6,73 "
        "bodu z piatich zápasov, priemerný počet strelených gólov 1,45 a "
        "inkasovaných 1,48. Priemerný počet dní od posledného zápasu je 8, "
        "pričom hodnota je obmedzená na 30 dní, aby extrémne prestávky "
        "nerozkolísali model. Elo ratingy sa počas piatich sezón pohybujú "
        "približne medzi 1 360 a 1 700 bodmi."
    )
    builder.paragraph(
        "Graf 3 ukazuje, ktoré atribúty náhodný les považuje za najdôležitejšie. "
        "Na prvých miestach sú Elo rating súpera, forma z posledných zápasov a "
        "strely na bránu, teda práve tie ukazovatele, ktoré opisujú aktuálnu "
        "silu tímu. To je v súlade s očakávaním a zároveň dobrá kontrola, že "
        "feature engineering nepriniesol náhodné alebo podozrivé atribúty."
    )
    builder.figure(FIGURES / "05_feature_importance.png", "Graf 3: Dôležitosť atribútov v náhodnom lese", 12.5)

    builder.heading("4.3 Porovnanie modelov", 2)
    builder.paragraph(
        "Tabuľka 4 porovnáva všetky modely a referencie na testovacej sezóne. "
        "Najvyššiu presnosť dosiahol gradient boosting (49,2 %), tesne pred "
        "náhodným lesom (47,9 %) a kurzovým favoritom (48,9 %). Logistická "
        "regresia mala nižšiu presnosť (44,2 %), ale najvyššie macro F1 "
        "(0,423), pretože ako jediný model aspoň občas tipol remízu."
    )

    rows = []
    for row in comparison.itertuples(index=False):
        rows.append(
            [
                row.model,
                pct(row.accuracy),
                num(row.balanced_accuracy, 3),
                num(row.f1_macro, 3),
                num(row.log_loss, 3),
                num(row.brier, 3),
            ]
        )
    builder.table(
        "Tab. 4: Porovnanie modelov na testovacej sezóne",
        ["Model", "Presnosť", "Vyváž. presnosť", "Macro F1", "Log loss", "Brier"],
        rows,
        widths=[4.8, 2.1, 2.7, 1.9, 1.8, 1.6],
    )
    builder.paragraph(
        "Všetky tri modely prekonali obidve jednoduché referencie. Voči trhu "
        "však zaostávajú: najlepší log loss medzi modelmi má náhodný les "
        "(1,035) oproti 1,019 pri kurzoch, teda rozdiel 0,017. Pri presnosti "
        "je gradient boosting dokonca o 0,3 percentuálneho bodu lepší než "
        "kurzový favorit, čo je však rozdiel v rámci štatistickej neistoty "
        "(pozri diskusiu)."
    )

    builder.paragraph(
        "Graf 4 ukazuje matice zámen. Náhodný les aj gradient boosting "
        "predpovedajú remízu v podstate nikdy – stĺpec pre remízu je v oboch "
        "maticiach nulový. Všetky remízy preto tipujú ako výhru jedného "
        "z tímov, väčšinou domácich. Logistická regresia remízy tipuje, no za "
        "cenu toho, že viac chybuje pri výhrach hostí. Tento kompromis "
        "vysvetľuje, prečo má najvyššie macro F1, ale nižšiu presnosť."
    )
    builder.figure(FIGURES / "03_confusion_matrices.png", "Graf 4: Matice zámen modelov a kurzového trhu", 15.0)

    builder.paragraph(
        "Tabuľka 5 rozkladá presnosť po triedach. Pri výhrach domácich sú "
        "stromové modely veľmi úspešné (úplnosť 0,790), pretože tipujú domácich "
        "často. Pri remízach majú úplnosť nula. Logistická regresia má pri "
        "remízach presnosť 0,271 a úplnosť 0,250, teda zachytí približne "
        "štvrtinu remíz, ale za cenu mnohých falošných poplachov."
    )
    per_class = load_tables()["per_class"]
    selected = per_class[per_class["model"].isin(
        ["Logistická regresia", "Náhodný les", "Gradient boosting"]
    )]
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            [
                row["model"],
                row["class"],
                num(row["precision"], 3),
                num(row["recall"], 3),
                num(row["f1"], 3),
                int(row["support"]),
            ]
        )
    builder.table(
        "Tab. 5: Metriky po jednotlivých triedach výsledku",
        ["Model", "Trieda", "Presnosť", "Úplnosť", "F1", "Počet"],
        rows,
        widths=[3.8, 4.8, 1.9, 1.7, 1.4, 1.3],
    )

    builder.figure(FIGURES / "04_calibration.png", "Graf 5: Kalibrácia pravdepodobností", 15.0)
    builder.paragraph(
        "Kalibrácia na grafe 5 ukazuje, že logistická regresia dáva mierne "
        "podhodnotené pravdepodobnosti pri výhrach domácich a naopak "
        "preceňuje remízy. Kurzy sú vo väčšine pásiem bližšie k ideálnej "
        "priamke. To je jeden z dôvodov, prečo má trh lepší log loss napriek "
        "podobnej presnosti."
    )

    builder.paragraph(
        "Tabuľka 6 ukazuje presnosť podľa toho, akú istotu model vyjadril. "
        "Vzťah je viditeľný najmä pri náhodnom lese: pri predikciách s istotou "
        "aspoň 60 % je presnosť 60,2 %, kým pri predikciách pod 40 % len "
        "29,3 %. Model teda vie aspoň približne rozlíšiť, kedy je jeho odhad "
        "spoľahlivejší. Pri logistickej regresii je tento vzťah slabší."
    )
    confidence = load_tables()["confidence"]
    pivot = confidence.pivot_table(
        index="model", columns="pasmo_istoty", values=["uspesnost", "pocet"], aggfunc="first"
    )
    rows = []
    for model in ["Logistická regresia", "Náhodný les", "Gradient boosting"]:
        row = [model]
        for band in ["< 40 %", "40–50 %", "50–60 %", "≥ 60 %"]:
            try:
                accuracy = pivot.loc[model, ("uspesnost", band)]
                count = pivot.loc[model, ("pocet", band)]
                row.append(f"{pct(accuracy)} ({int(count)})")
            except KeyError:
                row.append("–")
        rows.append(row)
    builder.table(
        "Tab. 6: Presnosť modelov podľa pásma istoty",
        ["Model", "< 40 %", "40–50 %", "50–60 %", "≥ 60 %"],
        rows,
        widths=[4.4, 2.7, 2.7, 2.7, 2.3],
    )

    builder.heading("4.4 Porovnanie s kurzami stávkovej kancelárie", 2)
    builder.paragraph(
        "Tabuľka 7 porovnáva modely s trhom podrobnejšie. Rozdiel v log loss "
        "oproti kurzom je najmenší pri náhodnom lese (0,017), pri gradient "
        "boostingu 0,021 a pri logistickej regresii 0,037. Modely sa s trhom "
        "zhodujú v 66–87 % tipov: logistická regresia sa odchyľuje najviac, "
        "pretože tipuje remízy, ktoré trh takmer nikdy neoznačí za favorita."
    )
    odds = load_tables()["odds"]
    rows = []
    for row in odds.itertuples(index=False):
        rows.append(
            [
                row.model,
                pct(row.accuracy),
                num(row.log_loss, 3),
                num(row.brier, 3),
                f"{row.delta_log_loss_vs_trh:+.3f}".replace(".", ","),
                pct(row.zhoda_s_favoritom_trhu),
            ]
        )
    builder.table(
        "Tab. 7: Porovnanie modelov s kurzami stávkovej kancelárie",
        ["Model", "Presnosť", "Log loss", "Brier", "Δ log loss", "Zhoda s trhom"],
        rows,
        widths=[4.4, 2.1, 1.8, 1.6, 2.3, 2.6],
    )
    builder.paragraph(
        "Priemerná istota modelov je 0,476 pri logistickej regresii, 0,534 pri "
        "náhodnom lese a 0,531 pri gradient boostingu; trh má 0,521. "
        "Logistická regresia je teda opatrnejšia a jej pravdepodobnosti sú "
        "bližšie k rovnomernejšiemu rozdeleniu, čo pri remízach pomáha, ale "
        "pri jasných favoritoch škodí."
    )

    builder.heading("4.5 Simulácia stávkovania (value bets)", 2)
    builder.paragraph(
        "Simulácia stávkovania ukazuje, či by model dokázal nájsť stávky "
        "s kladnou očakávanou hodnotou. Pri prahu 5 % stavil model na každú "
        "triedu, kde bola jeho pravdepodobnosť aspoň o päť percentuálnych "
        "bodov vyššia než pravdepodobnosť trhu. Výsledky sú v tabuľke 8 "
        "a na grafe 6."
    )
    value = load_tables()["value"]
    reference = ["Vždy výhra domácich", "Vždy remíza", "Vždy výhra hostí"]
    rows = []
    for row in value.itertuples(index=False):
        if row.model in reference:
            label = row.model
        elif row.hranica == 0.05:
            label = f"{row.model} (5 %)"
        else:
            continue
        low = "–" if pd.isna(row.roi_ci_low) else signed_pct(row.roi_ci_low)
        high = "–" if pd.isna(row.roi_ci_high) else signed_pct(row.roi_ci_high)
        rows.append(
            [
                label,
                str(int(row.pocet_stavok)),
                pct(row.uspesnost),
                signed_pct(row.roi),
                f"⟨{low}; {high}⟩",
            ]
        )
    builder.table(
        "Tab. 8: Simulácia value bets a referenčné stratégie",
        ["Stratégia", "Stávky", "Úspešnosť", "ROI", "95 % interval"],
        rows,
        widths=[4.9, 1.9, 2.3, 2.1, 3.6],
    )
    builder.paragraph(
        "Všetky tri modely dosiahli pri prahu 5 % kladné ROI medzi +3,2 % a "
        "+6,8 %. Interval spoľahlivosti však vo všetkých prípadoch zahŕňa "
        "nulu, takže výsledok nemožno považovať za preukázanú výhodu. "
        "Zaujímavý je kontext referenčných stratégií: slepé stávkovanie na "
        "remízy dosiahlo v sezóne 2025/26 ROI +7,7 %, teda viac než ktorýkoľvek "
        "model. Ukazuje to, aká veľká je náhodná zložka pri jednej sezóne."
    )
    builder.figure(FIGURES / "06_value_bets_roi.png", "Graf 6: Návratnosť simulovaných value bets", 13.0)

    builder.heading("4.6 Webová aplikácia", 2)
    builder.paragraph(
        "Aplikácia je naprogramovaná v knižnici Streamlit a spúšťa sa "
        "lokálne príkazom `streamlit run streamlit_app.py`. V bočnom paneli "
        "používateľ vyberie sezónu, domáci tím, hosťujúci tím a model. "
        "Aplikácia zobrazí pravdepodobnosti všetkých troch výsledkov, "
        "slovný tip a graf. Na obrázku 1 je predikcia zápasu Arsenal – Chelsea "
        "z pohľadu sezóny 2025/26."
    )
    builder.figure(FIGURES / "07_app_predikcia.jpg", "Obr. 1: Predikcia zápasu Arsenal – Chelsea v aplikácii", 15.0)
    builder.paragraph(
        "Predikcia vychádza zo stavu tímov pred ich posledným odohraným "
        "zápasom v zvolenej sezóne, pretože aplikácia nemá k dispozícii "
        "aktuálne dáta o zraneniach ani zostavách. Tento údaj je v aplikácii "
        "priamo uvedený, aby používateľ vedel, z akého obdobia model vychádza."
    )
    builder.paragraph(
        "Druhá karta ponúka porovnanie formy (obrázok 2). Zobrazuje výsledky "
        "posledných piatich zápasov, priemery gólov, strely na bránu, Elo "
        "rating, počet dní od posledného zápasu a vzájomnú bilanciu. Tretia "
        "karta zobrazuje metriky zvoleného modelu a jeho hyperparametre, aby "
        "bolo jasné, s akým modelom používateľ pracuje."
    )
    builder.figure(FIGURES / "08_app_porovnanie.jpg", "Obr. 2: Porovnanie formy tímov v aplikácii", 15.0)
    builder.paragraph(
        "Aplikácia je ošetrená aj proti chybovým stavom: ak chýbajú "
        "spracované dáta alebo model, zobrazí zrozumiteľnú hlášku s návodom, "
        "ako ich vytvoriť. Súčasťou projektu je aj automatický test, ktorý "
        "aplikáciu spustí bez prehliadača a overí, že sa zobrazí bez výnimky."
    )

    # ------------------------------------------------------------------
    # 5 DISKUSIA
    # ------------------------------------------------------------------
    builder.page_break()
    builder.heading("5 DISKUSIA")
    builder.paragraph(
        "Výsledky ukazujú, že modely sa z historických dát naučia užitočný "
        "signál: všetky tri prekonali náhodný tip aj pravidlo „vždy domáci“ "
        "a priblížili sa odhadu stávkového trhu. Zároveň však potvrdzujú, že "
        "futbal je náročná predikčná úloha. Rozdiel medzi modelmi je malý a "
        "pri 380 testovacích zápasoch ho treba čítať opatrne. Smerodajná "
        "odchýlka presnosti pri 380 zápasoch a skutočnej úspešnosti okolo "
        "50 % je približne 2,6 percentuálneho bodu. Rozdiel medzi 49,2 % "
        "a 48,9 % je teda hlboko pod hranicou štatistickej neistoty."
    )
    builder.paragraph(
        "Z tohto pohľadu je zaujímavejší rozdiel v log loss, ktorý hodnotí "
        "kvalitu pravdepodobností, nie iba počet správnych tipov. Kurzy majú "
        "log loss 1,019, najlepší model 1,035. Trh je teda presnejší, čo je "
        "prirodzené: do kurzov vstupujú informácie, ktoré v dátach nemáme – "
        "zranenia, zostavy, taktické zámery, počasie, motivácia či pohyb "
        "kurzov. Model pracuje len s výsledkami a základnými štatistikami."
    )
    builder.paragraph(
        "Najzaujímavejším zistením je správa modelov o remízach. Náhodný les "
        "a gradient boosting nepredpovedali remízu ani raz, hoci remízy "
        "tvoria 27,4 % testovacej sezóny. Je to dôsledok toho, že remíza "
        "nemá vlastný „signál“ v priemerných štatistikách – vzniká skôr "
        "z tesnosti zápasu. Logistická regresia remízy tipuje, ale s nízkou "
        "presnosťou. Ani jeden prístup nie je jednoznačne správny: stromové "
        "modely maximalizujú presnosť, logistická regresia zase vyváženosť "
        "tried. V praxi by preto dávalo zmysel hodnotiť model podľa účelu – "
        "iná metrika je vhodná na tipovanie víťaza a iná na odhad "
        "pravdepodobností."
    )
    builder.paragraph(
        "Voľba modelu podľa validačnej sezóny ukázala ďalší praktický problém. "
        "Najnižší validačný log loss mala logistická regresia (0,990), preto "
        "by bola vybraná ako finálny model. Na testovacej sezóne však mala "
        "najnižšiu presnosť a najvyšší log loss spomedzi modelov. Neznamená to "
        "chybu postupu – práve naopak, je to ukážka toho, že jediná sezóna "
        "je slabá vzorka a že výber podľa validácie nemusí platiť na nových "
        "dátach. Keby sme model vybrali podľa testu, dopustili by sme sa "
        "metodologickej chyby a výsledok by bol nadhodnotený."
    )
    builder.paragraph(
        "Simulácia stávkovania dopadla na prvý pohľad priaznivo – všetky "
        "modely mali pri prahu 5 % kladné ROI. Interval spoľahlivosti však "
        "v každom prípade zahŕňa nulu a šírka intervalu je viac ako 30 "
        "percentuálnych bodov. Navyše slepé stávkovanie na remízy dosiahlo "
        "vyššie ROI než modely, čo jasne ukazuje, že kladné ROI v jednej "
        "sezóne môže byť dielom náhody. Toto zistenie považujeme za dôležité "
        "a v práci ho neprezentujeme ako dôkaz ziskovej stratégie."
    )
    builder.paragraph(
        "Práca má niekoľko obmedzení. Prvým je rozsah dát: päť sezón a 1 900 "
        "zápasov stačí na trénovanie, ale nie na spoľahlivé rozlíšenie malých "
        "rozdielov medzi modelmi. Druhým je absencia informácií o zraneniach, "
        "zostavách a očakávaných góloch (xG); strely na bránu sú len "
        "približnou náhradou. Tretím je chýbajúca kalibrácia pravdepodobností "
        "– modely sme použili priamo, hoci metódy ako `CalibratedClassifierCV` "
        "by mohli log loss zlepšiť. Štvrtým je skutočnosť, že aplikácia "
        "pracuje so stavom tímov na konci sezóny, nie s aktuálnym stavom "
        "pred konkrétnym zápasom."
    )
    builder.paragraph(
        "Možné rozšírenia preto vidíme v štyroch smeroch: rozšírenie datasetu "
        "na desať a viac sezón a prípadne viac líg, doplnenie dát o xG a "
        "zranenia, kalibrácia pravdepodobností a testovanie ďalších modelov "
        "(napríklad Poissonových modelov na počet gólov). Zaujímavé by bolo "
        "aj porovnanie modelu, ktorý kurzy používa ako vstup – takéto riešenie "
        "by pravdepodobne dosiahlo lepší log loss, no menej by ukázalo, čo sa "
        "dá zistiť z čisto výkonnostných dát."
    )

    # ------------------------------------------------------------------
    # 6 ZÁVER
    # ------------------------------------------------------------------
    builder.page_break()
    builder.heading("6 ZÁVER")
    builder.paragraph(
        "Práca splnila stanovené ciele. Vznikol kompletný reťazec od "
        "stiahnutia historických dát cez ich čistenie a tvorbu atribútov až "
        "po trénovanie modelov, vyhodnotenie a webovú aplikáciu. Všetky kroky "
        "sa dajú spustiť jedným príkazom a sú pokryté automatickými testami, "
        "ktoré overujú najmä to, že atribúty neobsahujú informácie "
        "z budúcnosti."
    )
    builder.paragraph(
        "Na testovacej sezóne 2025/26 dosiahli modely presnosť 44,2 – 49,2 %, "
        "pričom kurzový favorit stávkových kancelárií mal 48,9 %. Všetky "
        "modely tak prekonali jednoduché referencie, ale neprekonali trh "
        "v kvalite pravdepodobností. Log loss najlepšieho modelu bol 1,035 "
        "oproti 1,019 pri kurzoch. Simulácia value bets síce priniesla kladné "
        "ROI, ale interval spoľahlivosti zahŕňal nulu, a preto ju "
        "nepovažujeme za dôkaz ziskovej stratégie."
    )
    builder.paragraph(
        "Prínos práce vidíme v tom, že spája dátovú analýzu, strojové učenie "
        "a webovú vizualizáciu do jedného funkčného celku a že výsledky "
        "neinterpretuje nekriticky. Ukazuje, kde sú hranice predikcie "
        "futbalových zápasov, prečo sú remízy problém a prečo je stávkový trh "
        "taký silný súper. Práca zároveň poskytuje otvorený základ, na ktorom "
        "sa dá ďalej stavať – rozšírením dát, kalibráciou alebo inými "
        "modelmi."
    )

    # ------------------------------------------------------------------
    # ZHRNUTIE + literatúra
    # ------------------------------------------------------------------
    builder.heading("Zhrnutie")
    builder.paragraph(
        "Táto práca sa zaoberá predikciou výsledkov futbalových stretnutí "
        "anglickej Premier League pomocou algoritmov strojového učenia. "
        "Vychádza z 1 900 zápasov z piatich sezón (2021/22 – 2025/26) a "
        "z 46 predzápasových atribútov, ktoré vznikli z výsledkov, striel, "
        "rohov a vzájomných zápasov. Dáta sú rozdelené chronologicky na "
        "trénovaciu, validačnú a testovaciu časť, pričom testovacia sezóna sa "
        "použila iba raz."
    )
    builder.paragraph(
        "Porovnávali sme logistickú regresiu, náhodný les a gradient "
        "boosting. Na testovacej sezóne dosiahol najvyššiu presnosť gradient "
        "boosting (49,2 %), náhodný les 47,9 % a logistická regresia 44,2 %; "
        "kurzový favorit stávkových kancelárií mal 48,9 %. Všetky modely "
        "prekonali náhodný tip aj pravidlo „vždy výhra domácich“, ale "
        "nedosiahli kvalitu pravdepodobností trhu. Ako problematické sa "
        "ukázali remízy, ktoré stromové modely netipujú vôbec."
    )
    builder.paragraph(
        "Pre interakciu s modelom bola vytvorená webová aplikácia v knižnici "
        "Streamlit, ktorá po výbere dvoch tímov zobrazí pravdepodobnosti "
        "výsledku, porovnanie formy a informácie o použitom modeli. Práca tak "
        "spája teoretické poznatky o klasifikácii s praktickou ukážkou "
        "použitia umelej inteligencie pri analýze športových dát."
    )

    builder.heading("Zoznam použitej literatúry")
    references = [
        "FOOTBALL-DATA.CO.UK. Historical Football Results and Betting Odds Data [online]. "
        "Dostupné na: https://www.football-data.co.uk",
        "GÉRON, Aurélien. 2022. Hands-On Machine Learning with Scikit-Learn, Keras and "
        "TensorFlow. 3. vydanie. Sebastopol: O'Reilly Media. ISBN 978-1-098-12597-4.",
        "MCKINNEY, Wes. 2022. Python for Data Analysis. 3. vydanie. Sebastopol: O'Reilly "
        "Media. ISBN 978-1-098-10403-0.",
        "PEDREGOSA, Fabian et al. 2011. Scikit-learn: Machine Learning in Python. "
        "Journal of Machine Learning Research, roč. 12, s. 2825–2830.",
        "DIXON, Mark J. – COLES, Stuart G. 1997. Modelling Association Football Scores and "
        "Inefficiencies in the Football Betting Market. Journal of the Royal Statistical "
        "Society: Series C, roč. 46, č. 2, s. 265–280.",
        "HVATTUM, Lars Magnus – ARNTZEN, Halvard. 2010. Using ELO ratings for match result "
        "prediction in association football. International Journal of Forecasting, "
        "roč. 26, č. 3, s. 460–470.",
        "ŠIOV. Metodicko-organizačné pokyny pre stredoškolskú odbornú činnosť. "
        "Bratislava: Štátny inštitút odborného vzdelávania.",
    ]
    for index, reference in enumerate(references, start=1):
        builder.paragraph(f"{index}. {reference}", indent=False, space_after=Pt(4))


# ---------------------------------------------------------------------------
# Prílohy
# ---------------------------------------------------------------------------
def build_appendices(document, anchor) -> None:
    from docx.shared import Pt

    builder = DocumentBuilder(document, anchor)

    builder.page_break()
    builder.heading("Príloha A: Zdrojový kód a elektronická príloha")
    builder.paragraph(
        "Zdrojový kód projektu je súčasťou elektronickej prílohy práce. "
        "Projekt je zorganizovaný do jedného priečinka `aplikacia`, ktorý "
        "obsahuje všetky skripty, konfiguráciu a testy. Kód nie je uvedený "
        "v tlačenej časti práce, pretože má približne 2 000 riadkov; v "
        "elektronickej prílohe je dostupný v pôvodnej podobe vrátane "
        "histórie zmien v systéme Git."
    )
    builder.paragraph("Štruktúra elektronickej prílohy:", space_after=Pt(3))
    structure = [
        "`src/config.py` – cesty, zoznam sezón, konštanty a nastavenie rozdelenia dát,",
        "`src/download.py` – sťahovanie CSV z football-data.co.uk,",
        "`src/prepare.py` – čistenie, zjednotenie názvov tímov a kontrola kvality,",
        "`src/features.py` – tvorba predzápasových atribútov a stavov tímov,",
        "`src/metrics.py` – spoločné metriky (presnosť, F1, log loss, Brier),",
        "`src/train.py` – ladenie hyperparametrov a trénovanie modelov,",
        "`src/evaluate.py` – vyhodnotenie, grafy a simulácia value bets,",
        "`streamlit_app.py` – webová aplikácia,",
        "`tests/` – automatické testy (ochrana proti úniku budúcnosti, test aplikácie),",
        "`data/`, `models/`, `reports/` – dáta, natrénované modely a výstupy,",
        "`run_all.ps1` – spustenie celého postupu jedným príkazom,",
        "`README.md` – inštalácia, spustenie a opis projektu.",
    ]
    for item in structure:
        builder.paragraph(f"• {item}", indent=False, space_after=Pt(2))

    builder.paragraph(
        "Postup spustenia: v priečinku `aplikacia` je potrebné vytvoriť "
        "virtuálne prostredie a nainštalovať závislosti z `requirements.txt`. "
        "Následne príkaz `run_all.ps1` stiahne dáta, vytvorí atribúty, "
        "natrénuje modely a vygeneruje tabuľky a grafy. Aplikácia sa spustí "
        "príkazom `streamlit run streamlit_app.py` a testy príkazom "
        "`pytest tests`. Podrobný opis je v súbore `README.md`."
    )
    builder.paragraph(
        "Projekt je možné otvoriť aj bez inštalácie prostredia: všetky "
        "výsledky uvedené v práci sú uložené v priečinku `reports` ako "
        "tabuľky (CSV, Markdown) a grafy (PNG, 300 dpi)."
    )

    builder.page_break()
    builder.heading("Príloha B: Snímky obrazovky z aplikácie Streamlit")
    builder.paragraph(
        "Na obrázku 1 je hlavná obrazovka aplikácie s predikciou zápasu "
        "Arsenal – Chelsea v sezóne 2025/26. Na obrázku 2 je karta "
        "s porovnaním formy oboch tímov."
    )
    builder.figure(FIGURES / "07_app_predikcia.jpg", "Obr. 1: Predikcia zápasu Arsenal – Chelsea v aplikácii", 15.5)
    builder.figure(FIGURES / "08_app_porovnanie.jpg", "Obr. 2: Porovnanie formy tímov v aplikácii", 15.5)


# ---------------------------------------------------------------------------
# Hlavný postup
# ---------------------------------------------------------------------------
def main() -> int:
    if not DOCX_PATH.exists():
        raise SystemExit(f"Chýba dokument: {DOCX_PATH}")

    document = Document(str(DOCX_PATH))
    strip_heading_numbering(document)
    fix_front_matter(document)
    fix_front_pagination(document)
    rebuild_abbreviations(document)
    rebuild_lists(document)
    insert_toc(document)

    anchor = clear_between(document, "ÚVOD", "PRÍLOHY")
    # Kapitoly musia zostať v tretej sekcii (tá má v pätičke číslo strany).
    # Posledný odsek sekcie nesie `sectPr`, preto vkladáme pred neho.
    paragraphs = document.paragraphs
    prilohy_index = next(
        i for i, p in enumerate(paragraphs) if p.text.strip() == "PRÍLOHY"
    )
    section_end = next(
        p
        for p in reversed(paragraphs[:prilohy_index])
        if "sectPr" in p._p.xml
    )
    build_chapters(DocumentBuilder(document, section_end))

    # Prílohy: odstránime pôvodné ukážkové odstavce za nadpisom PRÍLOHY.
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "PRÍLOHY")
    for paragraph in paragraphs[start + 1 :]:
        if "sectPr" in paragraph._p.xml:
            continue
        paragraph._p.getparent().remove(paragraph._p)

    tail = document.add_paragraph()
    build_appendices(document, tail)
    tail._p.getparent().remove(tail._p)

    enable_field_updates(document)
    document.save(str(DOCX_PATH))
    print(f"[ok] dokument uložený: {DOCX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
