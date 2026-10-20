"""The category taxonomy.

This file is the single source of truth for which categories exist.
The baseline, the LLM prompt, the evaluation and the API all import from here,
so adding or renaming a category happens in exactly one place.
"""

CATEGORIES: dict[str, str] = {
    "Groceries": "Supermarkets, grocery stores, food delivery of groceries.",
    "Restaurants": "Restaurants, cafes, bars, fast food, takeaway and food-delivery apps.",
    "Transport": "Public transport, taxis, ride-hailing, fuel, parking, bike and scooter hire.",
    "Shopping": "Clothes, electronics, department stores, online marketplaces, home goods.",
    "Entertainment": "Cinema, concerts, games, events, leisure activities, gyms and sport.",
    "Subscriptions": "Recurring digital services: streaming, software, cloud storage, memberships.",
    "Bills": "Utilities, phone and internet providers, insurance, rent, council tax.",
    "Travel": "Flights, hotels, holiday rentals, trains between cities, travel agencies.",
    "Health": "Pharmacies, doctors, dentists, opticians, health products.",
    "Transfers": "Money sent to or received from people, top-ups, account transfers.",
    "Cash": "ATM withdrawals and cash deposits.",
}

CATEGORY_NAMES: list[str] = list(CATEGORIES)

# Returned when a system cannot produce a valid category (e.g. broken LLM output).
# It is deliberately NOT a real category, so it always counts as an error in evaluation.
UNKNOWN = "Unknown"
