# Predikcia výsledkov futbalových zápasov

SOČ projekt (odbor 11 – Informatika) zameraný na predikciu výsledkov zápasov
anglickej Premier League pomocou strojového učenia a vizualizáciu predikcií
vo webovej aplikácii.

## Čo projekt robí

1. Stiahne historické výsledky a kurzy z [football-data.co.uk](https://www.football-data.co.uk).
2. Vyčistí dáta a vytvorí predzápasové atribúty (forma, góly, strely, Elo, H2H).
3. Natrénuje a porovná tri klasifikačné modely (logistická regresia, náhodný les,
   gradient boosting) s referenčnými modelmi a s kurzami stávkových kancelárií.
4. Vyhodnotí presnosť, log loss, kalibráciu a simuluje value bets.
5. Zobrazí predikciu ľubovoľnej dvojice tímov v Streamlit aplikácii.

## Požiadavky

- Windows 10/11, PowerShell
- Python 3.12 (nainštalovaný cez python.org alebo winget)
- ~2 GB miesta (virtuálne prostredie + knižnice)

## Inštalácia

```powershell
cd C:\Users\renej\Desktop\soc\aplikacia
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ak PowerShell blokuje aktiváciu prostredia:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Spustenie

Celý pipeline (sťahovanie → čistenie → atribúty → tréning → vyhodnotenie):

```powershell
.\run_all.ps1
```

Ak PowerShell skripty blokuje (chyba *running scripts is disabled on this system*):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
```

alebo trvalo:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Jednotlivé kroky:

```powershell
python -m src.download      # stiahne 5 sezón (existujúce preskočí)
python -m src.prepare       # vyčistí a zlúči dáta
python -m src.features      # vytvorí atribúty a stavy tímov
python -m src.train         # natrénuje a vyladí modely
python -m src.evaluate      # tabuľky a grafy
```

Aplikácia:

```powershell
streamlit run streamlit_app.py
```

Testy:

```powershell
pytest tests -q
```

## Štruktúra

```
aplikacia/
├── data/               # raw, interim a processed CSV (negenerované do Gitu)
├── src/
│   ├── config.py       # cesty, sezóny, konštanty, seed
│   ├── download.py     # sťahovanie z football-data.co.uk
│   ├── prepare.py      # čistenie a kontrola kvality
│   ├── features.py     # feature engineering bez úniku budúcnosti
│   ├── metrics.py      # spoločné metriky
│   ├── train.py        # ladenie a tréning modelov
│   └── evaluate.py     # vyhodnotenie, kalibrácia, value bets
├── models/             # natrénované .pkl modely + metadata.json
├── reports/
│   ├── figures/        # grafy 300 dpi pre prácu + screenshoty appky
│   └── tables/         # tabuľky a zhrnutia pre prácu
├── tests/              # pytest testy (ochrana proti leakage, smoke test appky)
├── .streamlit/         # konfigurácia témy Streamlit aplikácie
├── streamlit_app.py    # webová aplikácia
├── run_all.ps1         # celý pipeline jedným príkazom
└── requirements.txt
```

## Výsledky (testovacia sezóna 2025/26)

| Model | Accuracy | F1 macro | Log loss |
| :--- | ---: | ---: | ---: |
| Vždy výhra domácich (baseline) | 42,6 % | 0,199 | 19,814 |
| Náhodný tip (1/3) | 42,6 % | 0,199 | 1,099 |
| Kurzy Bet365 – favorit | 48,9 % | 0,365 | 1,019 |
| Logistická regresia | 44,2 % | 0,423 | 1,055 |
| Náhodný les | 47,9 % | 0,355 | 1,035 |
| **Gradient boosting** | **49,2 %** | 0,368 | 1,039 |

Presné čísla sa môžu mierne líšiť podľa verzií knižníc a použitého hardvéru.
Všetky tabuľky a grafy nájdeš v `reports/`.

## Metodika v skratke

- **Dáta:** 5 sezón Premier League (2021/22 – 2025/26), 1 900 zápasov.
- **Rozdelenie:** tréning 2021/22 – 2023/24, validácia 2024/25, test 2025/26.
- **Atribúty:** 46 predzápasových ukazovateľov (forma za 5 zápasov, domáca
  a vonkajšia bilancia, strely na bránu, Elo, vzájomné zápasy).
- **Ochrana proti leakage:** všetky kĺzavé štatistiky používajú `shift(1)`;
  Elo a H2H sa aktualizujú až po zápase. Overujú to testy v `tests/`.
- **Výber modelu:** podľa log loss na validačnej sezóne; test sa použije raz.

## Obmedzenia

- Model nepracuje so zraneniami, zostavami ani xG (dáta nie sú dostupné).
- Remízy sú ťažko predikovateľné; stromové modely ich takmer nikdy netipnú.
- Pozitívne ROI simulácie value bets je nutné brať opatrne – ide o jednu sezónu
  s veľkou varianciou (95 % interval spoľahlivosti zahŕňa nulu).
- Výsledok nie je stávková rada.

## Zdroj dát

FOOTBALL-DATA.CO.UK. *Historical Football Results and Betting Odds Data*
[online]. Dostupné na: <https://www.football-data.co.uk>.
