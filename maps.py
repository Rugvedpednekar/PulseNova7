import requests
import json

API_KEY = "AIzaSyA5YB0JWPxkgVQiLHHQAYbUx4ELb2H-V2M"

URL = "https://places.googleapis.com/v1/places:searchNearby"

# Example coordinates (Hartford, CT) — change if you want
LAT = 41.7637
LNG = -72.6851

payload = {
    "includedTypes": ["hospital"],   # try: ["doctor"]
    "maxResultCount": 5,
    "locationRestriction": {
        "circle": {
            "center": {"latitude": LAT, "longitude": LNG},
            "radius": 5000.0
        }
    }
}

headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    # FieldMask is REQUIRED
    "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount"
}

resp = requests.post(URL, headers=headers, json=payload, timeout=30)

print("Status:", resp.status_code)
try:
    data = resp.json()
except Exception:
    print("Non-JSON response:\n", resp.text)
    raise

# If error:
if resp.status_code != 200:
    print("Error response:")
    print(json.dumps(data, indent=2))
else:
    places = data.get("places", [])
    print(f"Found {len(places)} places:\n")
    for i, p in enumerate(places, 1):
        name = (p.get("displayName", {}) or {}).get("text", "Unknown")
        addr = p.get("formattedAddress", "No address")
        rating = p.get("rating", "NA")
        count = p.get("userRatingCount", "NA")
        loc = p.get("location", {})
        print(f"{i}. {name}")
        print(f"   {addr}")
        print(f"   rating: {rating} ({count})")
        print(f"   location: {loc}\n")
