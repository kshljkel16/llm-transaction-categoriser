"""Generate a synthetic dataset of messy card-transaction descriptions.

Why synthetic?
    Real bank transactions are private, and the public datasets are small or
    unlabelled. Generating data gives us exact ground truth (category AND the
    clean merchant name) and lets us control the difficulty. The cost: the
    "mess" is only as realistic as the rules below. GUIDE.md, Step 1, discusses
    this limitation — be ready to talk about it in an interview.

What makes it hard on purpose:
    1. Descriptors look like real card statements: upper case, payment-processor
       prefixes (SQ *, SUMUP *, PAYPAL *), store numbers, reference codes,
       city/country suffixes and truncation to ~22 characters.
    2. Merchant-level split: some merchants appear ONLY in val or ONLY in test.
       A model that memorises merchant names will fail on them. This mirrors
       production, where new merchants appear every day.

Usage:
    python -m txncat.generate_data            # writes data/{train,val,test}.csv
    python -m txncat.generate_data --seed 7   # different but reproducible data
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Merchant catalogue: category -> list of (canonical merchant name, descriptor templates)
# Template placeholders: {city} {store} {ref} {cc}
# ---------------------------------------------------------------------------
MERCHANTS: dict[str, list[tuple[str, list[str]]]] = {
    "Groceries": [
        ("Tesco", ["TESCO STORES {store}", "TESCO EXPRESS {city}", "TESCO.COM {ref}"]),
        ("Sainsbury's", ["SAINSBURYS S/MKT {city}", "SAINSBURY'S LOCAL {store}"]),
        ("Lidl", ["LIDL GB {city}", "LIDL {store} {city}"]),
        ("Aldi", ["ALDI {store} {city}", "ALDI STORES LTD"]),
        ("Waitrose", ["WAITROSE {store} {city}", "WAITROSE & PARTNERS"]),
        ("Co-op", ["CO-OP GROUP {store}", "COOP FOOD {city}"]),
        ("Morrisons", ["WM MORRISONS {store}", "MORRISONS {city}"]),
        ("Ocado", ["OCADO RETAIL {ref}", "OCADO.COM"]),
        ("Biedronka", ["BIEDRONKA {store} {city}", "JMP S.A. BIEDRONKA {store}"]),
        ("Carrefour", ["CARREFOUR MARKET {city}", "CARREFOUR {store}"]),
        ("Mercadona", ["MERCADONA {city}", "MERCADONA SA {store}"]),
        ("Pingo Doce", ["PINGO DOCE {city}", "PINGO DOCE {store}"]),
        ("M&S Food", ["M&S SIMPLY FOOD {store}", "MARKS&SPENCER FOOD {city}"]),
        ("Iceland", ["ICELAND FOODS {store}", "ICELAND {city}"]),
        ("Zabka", ["ZABKA {store} {city}", "ZABKA POLSKA {ref}"]),
    ],
    "Restaurants": [
        ("Pret A Manger", ["PRET A MANGER {city}", "PRET {store} {city}"]),
        ("Starbucks", ["STARBUCKS {store}", "STARBUCKS COFFEE {city}"]),
        ("Costa Coffee", ["COSTA COFFEE {store}", "COSTA {city}"]),
        ("McDonald's", ["MCDONALDS {store}", "MCDONALD'S {city}"]),
        ("Nando's", ["NANDOS {city}", "NANDO'S CHICKENLAND"]),
        ("Deliveroo", ["DELIVEROO {ref}", "DELIVEROO.CO.UK"]),
        ("Uber Eats", ["UBER *EATS {ref}", "UBER EATS {cc}"]),
        ("Just Eat", ["JUST EAT {ref}", "JUSTEAT.CO.UK"]),
        ("Wagamama", ["WAGAMAMA {city}", "WAGAMAMA LTD {store}"]),
        ("Greggs", ["GREGGS {store}", "GREGGS PLC {city}"]),
        ("Five Guys", ["FIVE GUYS {city}", "FIVE GUYS {store}"]),
        ("Itsu", ["ITSU {city}", "ITSU {store}"]),
        ("The Coffee House", ["SQ *THE COFFEE HOUSE", "SUMUP *COFFEE HOUSE"]),
        ("Kebab King", ["SUMUP *KEBAB KING", "SQ *KEBAB KING {city}"]),
        ("Glovo", ["GLOVO {ref}", "GLOVOAPP {city}"]),
        ("Leon", ["LEON RESTAURANTS {store}", "LEON {city}"]),
    ],
    "Transport": [
        ("TfL", ["TFL TRAVEL CH", "TFL.GOV.UK/CP", "TFL TRAVEL CHARGE"]),
        ("Uber", ["UBER *TRIP {ref}", "UBER *TRIP HELP.UBER.COM"]),
        ("Bolt", ["BOLT.EU/O/{ref}", "BOLT RIDE {city}"]),
        ("Shell", ["SHELL {city} {store}", "SHELL UK OIL"]),
        ("BP", ["BP {store} {city}", "BP CONNECT {city}"]),
        ("Lime", ["LIME*RIDE {ref}", "LIME RIDE {cc}"]),
        ("Santander Cycles", ["SANTANDER CYCLES", "TFL CYCLE HIRE"]),
        ("RingGo", ["RINGGO PARKING {ref}", "RINGGO {city}"]),
        ("FreeNow", ["FREENOW {ref}", "FREE NOW {city}"]),
        ("Orlen", ["ORLEN STACJA {store}", "PKN ORLEN {city}"]),
        ("Metro de Lisboa", ["METRO LISBOA {store}", "METROPOLITANO DE LISBOA"]),
        ("Addison Lee", ["ADDISON LEE {ref}", "ADDISONLEE LTD"]),
        ("NCP Parking", ["NCP LTD {city}", "NCP CAR PARK {store}"]),
    ],
    "Shopping": [
        ("Amazon", ["AMZN MKTP UK*{ref}", "AMAZON.CO.UK*{ref}", "AMZN MKTP {cc}"]),
        ("eBay", ["EBAY O*{ref}", "PAYPAL *EBAY"]),
        ("Zara", ["ZARA {city} {store}", "ZARA.COM {cc}"]),
        ("H&M", ["H&M {store} {city}", "HM.COM {ref}"]),
        ("IKEA", ["IKEA {city}", "IKEA LTD {store}"]),
        ("Argos", ["ARGOS LTD {store}", "ARGOS {city}"]),
        ("John Lewis", ["JOHN LEWIS {store}", "JOHNLEWIS.COM"]),
        ("Uniqlo", ["UNIQLO {city}", "UNIQLO EUROPE {store}"]),
        ("Apple Store", ["APPLE STORE R{store}", "APPLE STORE {city}"]),
        ("Currys", ["CURRYS {store}", "CURRYS PC WORLD {city}"]),
        ("Primark", ["PRIMARK {city}", "PRIMARK STORES {store}"]),
        ("ASOS", ["ASOS.COM {ref}", "ASOS {cc}"]),
        ("Vinted", ["VINTED {ref}", "VINTED UAB"]),
        ("TK Maxx", ["TK MAXX {store}", "TKMAXX {city}"]),
        ("Allegro", ["ALLEGRO.PL {ref}", "PAYU*ALLEGRO"]),
    ],
    "Entertainment": [
        ("Odeon", ["ODEON CINEMAS {city}", "ODEON {store}"]),
        ("Vue", ["VUE BSL LTD {city}", "VUE CINEMAS {store}"]),
        ("Ticketmaster", ["TICKETMASTER {ref}", "TICKETMASTER UK"]),
        ("Steam", ["STEAMGAMES.COM {ref}", "STEAM PURCHASE"]),
        ("PlayStation Store", ["PLAYSTATION NETWORK", "SONY INTERACTIVE {ref}"]),
        ("PureGym", ["PUREGYM {city}", "PUREGYM LTD"]),
        ("Eventbrite", ["EVENTBRITE/{ref}", "EB *{ref}"]),
        ("Hollywood Bowl", ["HOLLYWOOD BOWL {city}", "HOLLYWOOD BOWL GRP"]),
        ("Cineworld", ["CINEWORLD {city}", "CINEWORLD {store}"]),
        ("Dice", ["DICE FM {ref}", "DICE.FM"]),
        ("Nintendo eShop", ["NINTENDO CD{ref}", "NINTENDO ESHOP"]),
        ("Tate Modern", ["TATE ENTERPRISES", "TATE MODERN SHOP"]),
    ],
    "Subscriptions": [
        ("Netflix", ["NETFLIX.COM", "NETFLIX.COM {cc}"]),
        ("Spotify", ["SPOTIFY P{ref}", "SPOTIFY UK"]),
        ("Apple", ["APPLE.COM/BILL", "APPLE.COM/BILL {cc}"]),
        ("Google", ["GOOGLE *GOOGLE ONE", "GOOGLE *YOUTUBEPREMIUM"]),
        ("Disney+", ["DISNEYPLUS {ref}", "DISNEY PLUS"]),
        ("Amazon Prime", ["AMAZON PRIME*{ref}", "PRIME VIDEO {cc}"]),
        ("Microsoft", ["MICROSOFT*365 {ref}", "MSFT * {ref}"]),
        ("Adobe", ["ADOBE *CREATIVE CLD", "ADOBE {ref}"]),
        ("Dropbox", ["DROPBOX*{ref}", "DROPBOX"]),
        ("ChatGPT", ["OPENAI *CHATGPT SUBSCR", "OPENAI {ref}"]),
        ("Audible", ["AUDIBLE UK*{ref}", "AUDIBLE LTD"]),
        ("Duolingo", ["DUOLINGO {ref}", "DUOLINGO PLUS"]),
        ("NOW TV", ["NOWTV.COM {ref}", "NOW TV"]),
    ],
    "Bills": [
        ("British Gas", ["BRITISH GAS {ref}", "BRITISH GAS SERVICES"]),
        ("Octopus Energy", ["OCTOPUS ENERGY {ref}", "OCTOPUSENERGY"]),
        ("Thames Water", ["THAMES WATER {ref}", "THAMES WATER UTILITI"]),
        ("Vodafone", ["VODAFONE LTD {ref}", "VODAFONE {cc}"]),
        ("EE", ["EE LIMITED {ref}", "EE & T-MOBILE"]),
        ("BT", ["BT GROUP PLC {ref}", "BT.COM {ref}"]),
        ("Virgin Media", ["VIRGIN MEDIA {ref}", "VIRGIN MEDIA PYMTS"]),
        ("Council Tax", ["LB HACKNEY CTAX {ref}", "CITY COUNCIL C TAX"]),
        ("Aviva", ["AVIVA INSURANCE {ref}", "AVIVA LIFE"]),
        ("TV Licence", ["TV LICENCE MBP", "TVL {ref}"]),
        ("Giffgaff", ["GIFFGAFF {ref}", "GIFFGAFF.COM"]),
        ("EDP", ["EDP COMERCIAL {ref}", "EDP ENERGIAS"]),
        ("PGE", ["PGE OBROT {ref}", "PGE POLSKA"]),
    ],
    "Travel": [
        ("Ryanair", ["RYANAIR {ref}", "RYANAIR DAC"]),
        ("easyJet", ["EASYJET {ref}", "EASYJET AIRLINE"]),
        ("British Airways", ["BRITISH A {ref}", "BA.COM {ref}"]),
        ("Booking.com", ["BOOKING.COM {ref}", "HOTEL AT BOOKING.COM"]),
        ("Airbnb", ["AIRBNB * {ref}", "AIRBNB PAYMENTS"]),
        ("Trainline", ["TRAINLINE {ref}", "TRAINLINE.COM"]),
        ("Eurostar", ["EUROSTAR {ref}", "EUROSTAR INTL"]),
        ("Premier Inn", ["PREMIER INN {city}", "PREMIER INN {store}"]),
        ("Wizz Air", ["WIZZ AIR {ref}", "WIZZAIR.COM"]),
        ("Expedia", ["EXPEDIA {ref}", "EXPEDIA.CO.UK"]),
        ("TAP Air Portugal", ["TAP AIR PORTUGAL {ref}", "TAP PORTUGAL"]),
        ("Hilton", ["HILTON {city}", "HILTON HOTELS {store}"]),
        ("FlixBus", ["FLIXBUS {ref}", "FLIX SE {ref}"]),
    ],
    "Health": [
        ("Boots", ["BOOTS {store}", "BOOTS THE CHEMIST {city}"]),
        ("Superdrug", ["SUPERDRUG {store}", "SUPERDRUG {city}"]),
        ("Lloyds Pharmacy", ["LLOYDS PHARMACY {store}", "LLOYDSPHARMACY"]),
        ("Specsavers", ["SPECSAVERS {city}", "SPECSAVERS OPT {store}"]),
        ("Bupa", ["BUPA DENTAL {city}", "BUPA {ref}"]),
        ("Holland & Barrett", ["HOLLAND & BARRETT {store}", "HOLLAND+BARRETT {city}"]),
        ("Rossmann", ["ROSSMANN {store} {city}", "ROSSMANN SDP"]),
        ("Doctolib", ["DOCTOLIB {ref}", "DOCTOLIB SAS"]),
        ("Vision Express", ["VISION EXPRESS {city}", "VISION EXPRESS {store}"]),
        ("Farmacia", ["FARMACIA {city}", "FARMACIA CENTRAL {store}"]),
        ("Smile Dental", ["SUMUP *SMILE DENTAL", "SQ *SMILE DENTAL CLINIC"]),
    ],
}

# Transfers and cash have no real "merchant"; each tuple is a descriptor family.
TRANSFER_FAMILIES: list[tuple[str, list[str]]] = [
    ("", ["TO {person}", "TRANSFER TO {person}"]),
    ("", ["FROM {person}", "PAYMENT FROM {person}"]),
    ("", ["TOP-UP BY *{last4}", "APPLE PAY TOP-UP BY *{last4}"]),
    ("", ["TO SAVINGS POCKET", "TRANSFER TO SAVINGS"]),
    ("", ["REVOLUT USER {person}", "SENT TO {person}"]),
    ("", ["BANK TRANSFER {ref}", "FASTER PAYMENT {ref}"]),
]
CASH_FAMILIES: list[tuple[str, list[str]]] = [
    ("", ["ATM {bank} {city}", "CASH {bank} {city}"]),
    ("", ["CASH WITHDRAWAL {city}", "ATM WDL {store} {city}"]),
    ("", ["EURONET ATM {city}", "EURONET {store}"]),
    ("", ["CARDTRONICS {city}", "NOTEMACHINE {store}"]),
    ("", ["CASH DEPOSIT {city}", "PAYPOINT CASH DEP"]),
]

CITIES = ["LONDON", "MANCHESTER", "LEEDS", "BRISTOL", "EDINBURGH", "KRAKOW", "WARSZAWA",
          "LISBOA", "PORTO", "MADRID", "BARCELONA", "DUBAI", "GLASGOW", "BIRMINGHAM"]
COUNTRIES = ["GB", "GBR", "PL", "PT", "ES", "AE", "IE", "LU"]
PERSONS = ["J SMITH", "ANNA KOWALSKA", "M GARCIA", "OLIVER BROWN", "SOFIA COSTA", "A NOVAK",
           "DAVID LEE", "PRIYA PATEL", "T NGUYEN", "LUCAS SILVA", "E WILSON", "KAROLINA W"]
BANKS = ["HSBC", "BARCLAYS", "NATWEST", "LLOYDS", "SANTANDER", "PKO BP", "MILLENNIUM BCP"]
PROCESSOR_PREFIXES = ["SQ *", "SUMUP *", "PAYPAL *", "ZETTLE_*", "CRV*", "SP * "]

# Log-normal amount parameters per category: (median in GBP, spread)
AMOUNTS = {
    "Groceries": (25, 0.8), "Restaurants": (12, 0.6), "Transport": (8, 0.9),
    "Shopping": (35, 1.0), "Entertainment": (18, 0.7), "Subscriptions": (10, 0.5),
    "Bills": (60, 0.6), "Travel": (120, 0.9), "Health": (15, 0.8),
    "Transfers": (50, 1.2), "Cash": (40, 0.6),
}


@dataclass
class Row:
    description: str
    amount: float
    currency: str
    category: str
    merchant: str
    merchant_group: str          # id used for the merchant-level split
    merchant_seen_in_train: bool


def _ref(rng: random.Random) -> str:
    n = rng.randint(6, 10)
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=n))


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        city=rng.choice(CITIES),
        store=str(rng.randint(10, 9999)),
        ref=_ref(rng),
        cc=rng.choice(COUNTRIES),
        person=rng.choice(PERSONS),
        last4=str(rng.randint(1000, 9999)),
        bank=rng.choice(BANKS),
    )


def _add_noise(desc: str, rng: random.Random, allow_prefix: bool) -> str:
    """Make a descriptor look like a real card statement line."""
    if allow_prefix and rng.random() < 0.08 and not desc.startswith(("SQ *", "SUMUP", "PAYPAL")):
        desc = rng.choice(PROCESSOR_PREFIXES) + desc
    if rng.random() < 0.25:
        desc = f"{desc} {rng.choice(COUNTRIES)}"
    if rng.random() < 0.10:
        desc = desc.replace(" ", "", 1)               # glued words: "PRETA MANGER"
    if rng.random() < 0.30:
        desc = desc[: rng.randint(18, 25)].rstrip()   # card networks truncate descriptors
    if rng.random() < 0.05:
        desc = desc.title()                           # some acquirers send mixed case
    return " ".join(desc.split())


def _amount(category: str, desc: str, rng: random.Random) -> tuple[float, str]:
    median, spread = AMOUNTS[category]
    amount = math.exp(math.log(median) + rng.gauss(0, spread))
    if category == "Cash":
        amount = float(rng.choice([10, 20, 20, 40, 50, 100, 200]))
    currency = "GBP"
    if any(c in desc for c in ("KRAKOW", "WARSZAWA", "POLSKA", "BIEDRONKA", "ZABKA", "ORLEN", "PGE")):
        currency, amount = "PLN", amount * 5
    elif any(c in desc for c in ("LISBOA", "PORTO", "MADRID", "BARCELONA", "MERCADONA", "PINGO", "EDP")):
        currency, amount = "EUR", amount * 1.15
    elif "DUBAI" in desc:
        currency, amount = "AED", amount * 4.6
    return round(amount, 2), currency


def _split_groups(groups: list[str], rng: random.Random) -> dict[str, str]:
    """Assign each merchant group to 'train', 'val_only' or 'test_only' (~60/20/20)."""
    shuffled = groups[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, round(n * 0.2))
    n_test = max(1, round(n * 0.2))
    out = {}
    for i, g in enumerate(shuffled):
        out[g] = "val_only" if i < n_val else "test_only" if i < n_val + n_test else "train"
    return out


def generate(seed: int = 42, n_train: int = 6000, n_val: int = 1500, n_test: int = 1500):
    rng = random.Random(seed)

    catalogue: dict[str, list[tuple[str, str, list[str]]]] = {}
    for cat, merchants in MERCHANTS.items():
        catalogue[cat] = [(f"{cat}:{name}", name, t) for name, t in merchants]
    catalogue["Transfers"] = [(f"Transfers:{i}", m, t) for i, (m, t) in enumerate(TRANSFER_FAMILIES)]
    catalogue["Cash"] = [(f"Cash:{i}", m, t) for i, (m, t) in enumerate(CASH_FAMILIES)]

    assignment: dict[str, str] = {}
    for cat, items in catalogue.items():
        assignment.update(_split_groups([g for g, _, _ in items], rng))

    # Category frequencies roughly like a personal account (groceries/restaurants most common)
    weights = {"Groceries": 18, "Restaurants": 20, "Transport": 14, "Shopping": 12,
               "Entertainment": 6, "Subscriptions": 7, "Bills": 5, "Travel": 4,
               "Health": 4, "Transfers": 7, "Cash": 3}
    cats, cat_w = list(weights), list(weights.values())

    def make_rows(n: int, allowed: set[str]) -> list[Row]:
        rows = []
        while len(rows) < n:
            cat = rng.choices(cats, cat_w)[0]
            pool = [it for it in catalogue[cat] if assignment[it[0]] in allowed]
            if not pool:
                continue
            group, name, templates = rng.choice(pool)
            desc = _add_noise(_fill(rng.choice(templates), rng), rng,
                              allow_prefix=cat not in ("Transfers", "Cash"))
            amount, currency = _amount(cat, desc, rng)
            rows.append(Row(desc, amount, currency, cat, name, group,
                            merchant_seen_in_train=assignment[group] == "train"))
        return rows

    # val/test: ~70% known merchants, ~30% merchants never seen in training
    train = make_rows(n_train, {"train"})
    val = make_rows(int(n_val * 0.7), {"train"}) + make_rows(n_val - int(n_val * 0.7), {"val_only"})
    test = make_rows(int(n_test * 0.7), {"train"}) + make_rows(n_test - int(n_test * 0.7), {"test_only"})
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "description", "amount", "currency", "category", "merchant",
                    "merchant_group", "merchant_seen_in_train"])
        for i, r in enumerate(rows):
            w.writerow([i, r.description, r.amount, r.currency, r.category, r.merchant,
                        r.merchant_group, int(r.merchant_seen_in_train)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    train, val, test = generate(seed=args.seed)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        write_csv(rows, args.out / f"{name}.csv")
        unseen = sum(not r.merchant_seen_in_train for r in rows)
        print(f"{name:5s}: {len(rows):5d} rows, {unseen:4d} from merchants unseen in train")


if __name__ == "__main__":
    main()
