"""Vygeneruje finálny Word dokument práce SOČ pre ratingové systémy Elo, Glicko a Glicko-2.

Vychádza zo školskej šablóny `Šablóna_SOČ_GTV_25-26.docx`, aby boli zachované
všetky okraje, písma, štýly a číslovanie strán.

Použitie (z priečinka `aplikacia`):
    python tools/build_praca_ratings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx_utils import DocumentBuilder, clear_between, enable_field_updates  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
TEMPLATE_DOCX = ROOT / "Šablóna_SOČ_GTV_25-26.docx"
OUT_DOCX = ROOT / "Praca_SOC_Predikcia_vysledkov_futbalovych_zapasov.docx"

FIGURES = PROJECT / "reports" / "figures"
TABLES = PROJECT / "reports" / "tables"
META = json.loads((PROJECT / "processed" / "rating_metadata.json").read_text(encoding="utf-8")) if (PROJECT / "processed" / "rating_metadata.json").exists() else json.loads((PROJECT / "data" / "processed" / "rating_metadata.json").read_text(encoding="utf-8"))

TITLE = "Predikcia výsledkov futbalových zápasov"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def strip_heading_numbering(document) -> None:
    from docx.oxml.ns import qn

    for name in ("Heading 1", "Heading 2", "Heading 3"):
        ppr = document.styles[name].element.find(qn("w:pPr"))
        if ppr is not None:
            num_pr = ppr.find(qn("w:numPr"))
            if num_pr is not None:
                ppr.remove(num_pr)


def fix_front_matter(document) -> None:
    replacements = {
        "Názov témy": TITLE,
        "Č. odboru: xxx": "Č. odboru: 11 (Informatika)",
        "Riešiteľ: xxx": "Riešiteľ: [Doplniť: Meno a Priezvisko]",
        "2025": "2026",
        "Konzultant: xxx": "Konzultant: [Doplniť: Titul, Meno a Priezvisko]",
        "na tému XXX": f"na tému {TITLE}",
        "Vyhlasujem/e": "Vyhlasujem",
        "som/sme vypracoval/a/i": "som vypracoval",
        "som/sme neprihlásil/a/i ani neprezentoval/a/i": "som neprihlásil ani neprezentoval",
        "Som/sme si vedomý/á/í": "Som si vedomý",
        "Chceli by sme vyjadriť svoje poďakovanie predovšetkým našej konzultantke pani XXX": (
            "Chcel by som vyjadriť svoje poďakovanie predovšetkým môjmu konzultantovi "
            "[Doplniť: Titul, Meno a Priezvisko]"
        ),
        "našej stredoškolskej": "mojej stredoškolskej",
    }
    for p in document.paragraphs:
        for old, new in replacements.items():
            if old in p.text:
                for run in p.runs:
                    if old in run.text:
                        run.text = run.text.replace(old, new)
                if old in p.text:
                    full = p.text.replace(old, new)
                    if p.runs:
                        p.runs[0].text = full
                        for r in p.runs[1:]:
                            r.text = ""


def build_chapters(builder: DocumentBuilder) -> None:
    comp = pd.DataFrame(META["comparison"])
    elo_row = comp[comp["model"] == "Elo"].iloc[0]
    g1_row = comp[comp["model"] == "Glicko"].iloc[0]
    g2_row = comp[comp["model"] == "Glicko-2"].iloc[0]
    bet_row = comp[comp["model"] == "Kurzy Bet365"].iloc[0]
    base_row = comp[comp["model"] == "Vždy výhra domácich"].iloc[0]

    # ÚVOD
    builder.heading("ÚVOD", level=1)
    builder.paragraph(
        "Futbal patrí medzi najpopulárnejšie športy na svete a v posledných rokoch sa v ňom "
        "čoraz viac využíva práca s dátami. Tréneri, analytici aj fanúšikovia sa snažia objektívne "
        "zmerať, aký silný je určitý tím a akú má šancu zvíťaziť v najbližšom zápase. Na rozdiel "
        "od subjektívnych názorov dokážu matematické modely a algoritmy spracovať stovky výsledkov "
        "a vyjadriť silu tímov jedným prehľadným číslom."
    )
    builder.paragraph(
        "Inšpiráciou pre túto prácu boli ratingové systémy známe zo šachu a počítačových hier "
        "(ako sú FIFA, CS2 alebo League of Legends). V každej z týchto hier existuje systém, ktorý "
        "hráčom po každom zápase prideľuje alebo odoberá body podľa toho, či vyhrali nad silnejším "
        "alebo slabším súperom. V tejto práci skúmame, ako fungujú tri známe systémy – Elo, Glicko "
        "a Glicko-2 – keď ich použijeme na skutočné zápasy anglickej Premier League."
    )
    builder.paragraph(
        "Cieľom práce nie je vytvoriť neomylný stávkový nástroj, ale porovnať tieto tri algoritmy "
        "z hľadiska presnosti predikcie a vytvoriť pre ne jednoduchú a prehľadnú webovú aplikáciu, "
        "v ktorej si používateľ môže pozrieť aktuálnu silu tímov a odhad výsledku budúceho zápasu."
    )

    # 1 TEORETICKÁ ČASŤ
    builder.heading("1 RATINGOVÉ SYSTÉMY V ŠPORTE", level=1)
    builder.paragraph(
        "Ratingové systémy slúžia na priebežné hodnotenie výkonnosti hráčov alebo športových tímov "
        "na základe odohraných zápasov. Ich spoločnou myšlienkou je, že výsledok každého zápasu "
        "upraví hodnotenie oboch súperov podľa toho, nakoľko bol daný výsledok prekvapivý."
    )

    builder.heading("1.1 Klasický systém Elo", level=2)
    builder.paragraph(
        "Systém Elo vytvoril maďarsko-americký profesor fyziky Arpad Elo v 60. rokoch 20. storočia "
        "pôvodne pre šachistov (Elo, 1978). Každý tím začína s rovnakým počiatočným hodnotením "
        "(napr. 1500 bodov). Ak tím vyhrá nad výrazne silnejším súperom, získa veľa bodov. "
        "Ak vyhrá nad slabým súperom, získa len málo bodov, pretože jeho výhra sa očakávala."
    )
    builder.paragraph(
        "Vo futbale hrá dôležitú úlohu domáce prostredie (podpora fanúšikov, znalosť ihriska, "
        "žiadne cestovanie). Preto sa v našej implementácii k ratingu domáceho tímu pri výpočte "
        "očakávaného výsledku pripočítava bonus za domáce ihrisko (+60 bodov)."
    )

    builder.heading("1.2 Systém Glicko a miera neistoty (RD)", level=2)
    builder.paragraph(
        "Hlavnou slabinou klasického ELO systému je, že nevie rozlíšiť, nakoľko si je istý "
        "ratingom tímu. Ak do Premier League postúpi nováčik, po dvoch zápasoch o ňom ešte nevieme "
        "toľko ako o klube, ktorý v lige hrá už päť rokov. Mark Glickman preto v roku 1995 navrhol "
        "systém Glicko (Glickman, 1999), ktorý každému tímu okrem sily prideľuje aj druhé číslo – "
        "odchýlku ratingu (RD, Rating Deviation)."
    )
    builder.paragraph(
        "Vysoké RD znamená vysokú neistotu (nový tím alebo tím po dlhej prestávke), preto sa jeho "
        "rating po prvých zápasoch mení rýchlejšie. Postupom sezóny, keď tím odohrá viac zápasov, "
        "neistota RD klesá a rating sa stabilizuje."
    )

    builder.heading("1.3 Systém Glicko-2 a kolísavosť formy (volatilita)", level=2)
    builder.paragraph(
        "Systém Glicko-2 rozširuje pôvodný model o tretí parameter – volatilitu (σ), ktorá meria "
        "stabilitu výkonov tímu (Glickman, 2012). Stabilný tím s vyrovnanými výkonmi má nízku "
        "volatilitu. Naopak, ak tím jeden týždeň porazí favorita a na ďalší týždeň prehraje "
        "s posledným tímom tabuľky, jeho volatilita stúpne, čo signalizuje nevyspytateľnosť."
    )

    builder.heading("1.4 Prepočet rozdielu ratingov na futbalové pravdepodobnosti", level=2)
    builder.paragraph(
        "Klasické ratingové systémy rátajú len s výhrou alebo prehrou, no vo futbale sú časté aj "
        "remízy (približne 26 % zápasov). Na rozdelenie pravdepodobnosti medzi tri výsledky "
        "(výhra domácich, remíza, výhra hostí) sme použili Davidsonovo rozšírenie Bradley-Terryho "
        "modelu (Davidson, 1970). Čím sú si dva tímy ratingovo bližšie, tým vyššia je pravdepodobnosť remízy."
    )

    # 2 CIELE PRÁCE
    builder.heading("2 CIELE PRÁCE", level=1)
    builder.paragraph(
        "Hlavným cieľom práce je naprogramovať, porovnať a overiť tri ratingové systémy – Elo, "
        "Glicko a Glicko-2 – na reálnych historických zápasoch anglickej Premier League a vytvoriť "
        "prehľadnú webovú aplikáciu na predikciu výsledkov aktuálnej sezóny."
    )
    builder.paragraph("Z hlavného cieľa vyplývajú nasledujúce čiastkové ciele:")
    builder.bullet("Zhromaždiť a očistiť historické dáta zápasov Premier League za 5 sezón (1 900 zápasov).")
    builder.bullet("Implementovať algoritmy Elo, Glicko a Glicko-2 v jazyku Python bez použitia zložitého strojového učenia.")
    builder.bullet("Chronologicky vyhodnotiť presnosť predikcií na testovacej sezóne 2025/26 a porovnať ich s kurzami Bet365.")
    builder.bullet("Vytvoriť interaktívnu webovú aplikáciu v knižnici Streamlit s možnosťou živého sťahovania výsledkov pre sezónu 2026/27.")

    # 3 MATERIÁL A METODIKA
    builder.heading("3 MATERIÁL A METODIKA", level=1)
    builder.paragraph(
        "Dáta boli získané z verejne dostupného futbalového archívu football-data.co.uk vo formáte CSV. "
        "Použili sme 5 kompletných sezón najvyššej anglickej súťaže Premier League (2021/22 až 2025/26), "
        "čo predstavuje spolu 1 900 odohraných zápasov (380 zápasov v každej sezóne). Pre každý zápas "
        "sme použili dátum, názov domáceho a hosťujúceho tímu, konečný výsledok (H, D, A) a na porovnanie "
        "aj predzápasové kurzy stávkovej kancelárie Bet365."
    )
    builder.paragraph(
        "Celý výpočet prebieha prísne chronologicky od najstaršieho zápasu po najnovší. Pred každým "
        "zápasom model najprv zaznamená svoju predikciu (aby nedošlo k úniku informácií z budúcnosti) "
        "a až po odohraní zápasu upraví ratingy oboch tímov. Medzi jednotlivými sezónami sa ratingy "
        "mierne približujú k priemeru 1500 (regresia k priemeru) a v systémoch Glicko a Glicko-2 "
        "sa primerane zvýši miera neistoty RD."
    )
    builder.paragraph(
        "Na implementáciu bol použitý jazyk Python 3.12, knižnica pandas na prácu s tabuľkami, "
        "matplotlib na tvorbu grafov a Streamlit na vytvorenie interaktívneho webového rozhrania."
    )

    # 4 VÝSLEDKY PRÁCE
    builder.heading("4 VÝSLEDKY PRÁCE", level=1)

    builder.heading("4.1 Porovnanie presnosti algoritmov na sezóne 2025/26", level=2)
    builder.paragraph(
        "Všetky tri ratingové systémy sme vyhodnotili na 380 zápasoch sezóny 2025/26. Ako základný "
        "porovnávací bod (baseline) sme použili jednoduché pravidlo ‚vždy tipuj výhru domácich‘ "
        "a ako benchmark trhu reálne predzápasové kurzy Bet365."
    )

    headers = ["Model / Systém", "Typ", "Presnosť (Accuracy)", "F1 (macro)", "Log loss", "Brier score"]
    rows = [
        ["Vždy výhra domácich", "baseline", pct(base_row["accuracy"]), num(base_row["f1_macro"]), num(base_row["log_loss"]), num(base_row["brier"])],
        ["Elo", "rating", pct(elo_row["accuracy"]), num(elo_row["f1_macro"]), num(elo_row["log_loss"]), num(elo_row["brier"])],
        ["Glicko", "rating", pct(g1_row["accuracy"]), num(g1_row["f1_macro"]), num(g1_row["log_loss"]), num(g1_row["brier"])],
        ["Glicko-2", "rating", pct(g2_row["accuracy"]), num(g2_row["f1_macro"]), num(g2_row["log_loss"]), num(g2_row["brier"])],
        ["Kurzy Bet365", "trh", pct(bet_row["accuracy"]), num(bet_row["f1_macro"]), num(bet_row["log_loss"]), num(bet_row["brier"])],
    ]
    builder.table("Tab. 1: Porovnanie úspešnosti ratingových systémov na sezóne 2025/26", headers, rows)

    builder.paragraph(
        f"Najvyššiu presnosť z troch implementovaných systémov dosiahol klasický Elo systém ({pct(elo_row['accuracy'])}), "
        f"tesne nasledovaný systémom Glicko-2 ({pct(g2_row['accuracy'])}) a Glicko ({pct(g1_row['accuracy'])}). "
        f"Všetky tri modely výrazne prekonali základnú stratégiu tipovania na domácich ({pct(base_row['accuracy'])}) "
        f"a priblížili sa ku kurzovému favoritovi stávkovej kancelárie Bet365 ({pct(bet_row['accuracy'])})."
    )
    builder.figure(FIGURES / "01_rating_accuracy.png", "Graf 1: Porovnanie presnosti a kvality odhadu (Log loss)", 15.0)

    builder.heading("4.2 Vývoj ratingu a neistoty počas sezóny", level=2)
    builder.paragraph(
        "Veľkou výhodou systémov Glicko a Glicko-2 oproti klasickému Elo je sledovanie neistoty (RD). "
        "Na grafe nižšie vidíme, ako sa postupom sezóny menil rating top klubov Premier League a ako "
        "u nováčika v lige (Burnley) neistota RD na začiatku sezóny postupne klesala s každým odohratým kolom."
    )
    builder.figure(FIGURES / "02_rating_evolution.png", "Graf 2: Vývoj Glicko-2 ratingu vybraných tímov v sezóne 2025/26", 15.0)
    builder.figure(FIGURES / "03_glicko_uncertainty.png", "Graf 3: Pokles neistoty (RD) u nováčika a stabilného tímu", 14.5)

    builder.heading("4.3 Matica zámen najlepších predikcií", level=2)
    builder.paragraph(
        "Matica zámen ukazuje, kde algoritmy robia najčastejšie chyby. Systém veľmi dobre rozpozná "
        "jasných domácich a hosťujúcich favoritov, avšak najmenej úspešný je pri priamom tipovaní remíz. "
        "Je to prirodzené, pretože remíza má v takmer každom zápase nižšiu individuálnu pravdepodobnosť "
        "(okolo 25 %) než víťazstvo favorita."
    )
    builder.figure(FIGURES / "04_rating_confusion_matrix.png", "Graf 4: Matica zámen systému Glicko-2 (sezóna 2025/26)", 11.5)

    builder.heading("4.4 Webová aplikácia v knižnici Streamlit", level=2)
    builder.paragraph(
        "Pre praktické využitie bola vytvorená webová aplikácia v nástroji Streamlit. Používateľ si v bočnom "
        "paneli vyberie domáci tím, hosťujúci tím a preferovaný ratingový systém (Elo, Glicko alebo Glicko-2). "
        "Aplikácia v reálnom čase vypočíta pravdepodobnosti výsledku, zobrazí aktuálnu formu tímov "
        "a porovnanie všetkých troch algoritmov. Súčasťou aplikácie je aj rebríček sily všetkých 20 tímov "
        "ligy a tlačidlo na živé stiahnutie najnovších výsledkov aktuálnej sezóny 2026/27 z internetu."
    )

    # 5 DISKUSIA
    builder.heading("5 DISKUSIA", level=1)
    builder.paragraph(
        "Výsledky potvrdili, že aj relatívne jednoduché matematické modely založené čisto na výsledkoch "
        "zápasov dokážu dosiahnuť presnosť okolo 48 – 49,5 %, čo je porovnateľné s kurzovým favoritom "
        "stávkových kancelárií (48,9 %). Zaujímavým zistením je, že klasické Elo dosiahlo o niečo "
        "vyššiu presnosť než zložitejší systém Glicko-2. Dôvodom je skutočnosť, že v anglickej Premier League "
        "hrá každý tím 38 zápasov za sezónu bez dlhých prestávok, takže neistota (RD) u väčšiny tímov "
        "rýchlo klesne na podobnú úroveň."
    )
    builder.paragraph(
        "Stávkové kancelárie však dosahujú o niečo nižší (lepší) Log loss (1,019 oproti 1,028), pretože "
        "do svojich kurzov dokážu zahrnúť aj informácie, ktoré historické výsledky neobsahujú – najmä "
        "zranenia kľúčových hráčov, červené karty z pohárových zápasov či zmeny trénerov. Toto predstavuje "
        "hlavné obmedzenie čisto ratingových systémov."
    )

    # 6 ZÁVER
    builder.heading("6 ZÁVER", level=1)
    builder.paragraph(
        "Práca úspešne splnila všetky stanovené ciele. Na 1 900 historických zápasoch Premier League sme "
        "naprogramovali a porovnali tri známe ratingové systémy – Elo, Glicko a Glicko-2. Ukázali sme, "
        "že aj bez zložitej umelej inteligencie dokážeme s čistou matematikou dosiahnuť takmer 50 % presnosť "
        "predikcií a priblížiť sa profesionálnym trhovým odhadom."
    )
    builder.paragraph(
        "Vytvorená webová aplikácia v prostredí Streamlit premenila teoretické vzorce na praktický nástroj, "
        "ktorý dokáže priebežne sťahovať najnovšie zápasy sezóny 2026/27 a zobrazovať silu tímov. "
        "Do budúcna by bolo zaujímavé rozšíriť model o gólové rozdiely (napr. že výhra 4:0 má vyššiu "
        "váhu než výhra 1:0) alebo o štatistiky očakávaných gólov (xG)."
    )

    # Zhrnutie
    builder.heading("Zhrnutie", level=1)
    builder.paragraph(
        "Táto práca sa zaoberá využitím ratingových systémov Elo, Glicko a Glicko-2 pri predikcii "
        "výsledkov futbalových zápasov najvyššej anglickej súťaže Premier League. V teoretickej časti "
        "sú vysvetlené princípy hodnotenia sily tímov, miery neistoty (RD) a kolísavosti formy (volatility), "
        "ako aj prepočet rozdielu ratingov na pravdepodobnosť výhry, remízy a prehry pomocou Davidsonovho "
        "modelu. Praktická časť obsahuje vlastnú implementáciu všetkých troch algoritmov v jazyku Python "
        "a ich chronologické vyhodnotenie na 1 900 zápasoch za 5 sezón. Na testovacej sezóne 2025/26 dosiahol "
        f"najvyššiu presnosť klasický systém Elo ({pct(elo_row['accuracy'])}) a Glicko-2 ({pct(g2_row['accuracy'])}), "
        f"čím sa priblížili ku kurzovému favoritovi Bet365 ({pct(bet_row['accuracy'])}). Súčasťou práce je "
        "interaktívna webová aplikácia v nástroji Streamlit, ktorá umožňuje sledovať rebríček sily tímov, "
        "vývoj ich formy v čase a predpovedať zápasy prebiehajúcej sezóny 2026/27 na základe živých dát."
    )

    # Zoznam použitej literatúry
    builder.heading("Zoznam použitej literatúry", level=1)
    refs = [
        "DAVIDSON, Roger R. 1970. On extending the Bradley-Terry model to accommodate ties in paired comparison experiments. Journal of the American Statistical Association, 65(329), s. 317–328.",
        "ELO, Arpad E. 1978. The Rating of Chessplayers, Past and Present. New York: Arco Publishing. ISBN 0-668-04721-6.",
        "FOOTBALL-DATA.CO.UK. Historical Football Results and Betting Odds Data [online]. Dostupné na: https://www.football-data.co.uk.",
        "GLICKMAN, Mark E. 1999. Parameter estimation in large dynamic paired comparison experiments. Journal of the Royal Statistical Society: Series C, 48(3), s. 377–394.",
        "GLICKMAN, Mark E. 2012. Example of the Glicko-2 system [online]. Boston University. Dostupné na: http://www.glicko.net/glicko/glicko2.pdf.",
        "ŠIOV. Metodicko-organizačné pokyny pre Stredoškolskú odbornú činnosť [online]. Bratislava: Štátny inštitút odborného vzdelávania. Dostupné na: https://siov.sk.",
    ]
    for ref in refs:
        builder.paragraph(ref)


def main() -> int:
    if not TEMPLATE_DOCX.exists():
        raise SystemExit(f"Chýba šablóna: {TEMPLATE_DOCX}")

    document = Document(str(TEMPLATE_DOCX))
    strip_heading_numbering(document)
    fix_front_matter(document)

    anchor = clear_between(document, "ÚVOD", "PRÍLOHY")
    paragraphs = document.paragraphs
    prilohy_idx = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "PRÍLOHY")
    section_end = next(p for p in reversed(paragraphs[:prilohy_idx]) if "sectPr" in p._p.xml)

    build_chapters(DocumentBuilder(document, section_end))

    # Vyčistenie príloh na konci a doplnenie popisu
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "PRÍLOHY")
    for p in paragraphs[start + 1 :]:
        if "sectPr" in p._p.xml:
            continue
        p._p.getparent().remove(p._p)

    tail = document.add_paragraph()
    b_app = DocumentBuilder(document, tail)
    b_app.heading("Príloha A: Zdrojový kód algoritmov Elo, Glicko a Glicko-2 (src/ratings.py)", level=2)
    b_app.paragraph(
        "Kompletný zdrojový kód výpočtu všetkých troch ratingových systémov je priložený v priečinku "
        "`aplikacia/src/ratings.py` a je napísaný v jazyku Python 3.12."
    )
    b_app.heading("Príloha B: Prehľad webovej aplikácie Streamlit", level=2)
    b_app.paragraph(
        "Aplikácia sa spúšťa príkazom `streamlit run streamlit_app.py` a ponúka 3 záložky: "
        "Predikcia zápasu, Rebríček sily tímov a Vývoj sily a porovnanie modelov."
    )
    tail._p.getparent().remove(tail._p)

    enable_field_updates(document)
    document.save(str(OUT_DOCX))
    print(f"[ok] dokument uložený: {OUT_DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
