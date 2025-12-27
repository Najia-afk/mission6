"""
Simple OpenFoodFacts v1 (CGI) search.
Fetch products whose ingredients contain a given term (--ingr).
Outputs columns: foodId,label,category,foodContentsLabel,image.

Endpoint: https://world.openfoodfacts.org/cgi/search.pl

Examples:
  python fetch_openfoodfacts.py --ingr champagne -n 10
  python fetch_openfoodfacts.py --ingr champagne -q sparkling -n 10
  python fetch_openfoodfacts.py --ingr champagne -n 10 -o champagne_products.csv
"""
import argparse
import requests
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

SEARCH_V1_URL = "https://world.openfoodfacts.org/cgi/search.pl"
USER_AGENT = "Mission6Script/1.0 (macOS; +https://world.openfoodfacts.org)"
TIMEOUT = 25
FIELDS = [
    "code",
    "product_name",
    "generic_name",
    "categories",
    "categories_tags",
    "ingredients_text_en",
    "ingredients_text",
    "image_url",
    "image_front_url",
]

def fetch_openfoodfacts(limit: int,
                        ingredient: str,
                        query: str = None,
                        page_size: int = None,
                        all_pages: bool = False,
                        nocache: bool = False) -> pd.DataFrame:
    ingredient = ingredient.strip().lower()
    if page_size is None:
        page_size = min(limit, 100)
    page_size = max(1, min(page_size, 1000))

    collected = []
    page = 1
    headers = {"User-Agent": USER_AGENT}

    while len(collected) < limit:
        params = {
            "action": "process",
            "json": 1,
            "page": page,
            "page_size": page_size,
            "fields": ",".join(FIELDS),
            "tagtype_0": "ingredients",
            "tag_contains_0": "contains",
            "tag_0": ingredient,
        }
        if query:
            params["search_terms"] = query
            params["search_simple"] = 1
        if nocache:
            params["nocache"] = 1

        r = requests.get(SEARCH_V1_URL, params=params, timeout=TIMEOUT, headers=headers)
        data = r.json()
        products = data.get("products") or []
        if not products:
            break

        for p in products:
            ing_full = (p.get("ingredients_text_en") or p.get("ingredients_text") or "").strip()
            if ingredient not in ing_full.lower():
                continue
            label = (p.get("product_name") or p.get("generic_name") or "").strip()
            cats_tags = p.get("categories_tags") or []
            cats_str = p.get("categories") or ""
            if cats_tags:
                category = cats_tags[0].replace("en:", "")
            else:
                category = cats_str.split(",")[0].strip() if cats_str else ""
            image = p.get("image_url") or p.get("image_front_url") or ""
            collected.append({
                "foodId": p.get("code") or "",
                "label": label,
                "category": category,
                "foodContentsLabel": ing_full,
                "image": image
            })
            if len(collected) >= limit:
                break

        if not all_pages:
            break
        page += 1

    return pd.DataFrame(collected[:limit])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingr", required=True, help="Ingredient term (e.g. champagne)")
    ap.add_argument("--limit", "-n", type=int, default=10, help="Number of products")
    ap.add_argument("--query", "-q", help="Additional free-text narrowing (optional)")
    ap.add_argument("--output", "-o", help="CSV output path (optional)")
    ap.add_argument("--page-size", type=int, help="Custom page_size (default=min(limit,100))")
    ap.add_argument("--all-pages", action="store_true", help="Iterate pages until limit reached")
    ap.add_argument("--nocache", action="store_true", help="Add nocache=1 to bypass cache")
    args = ap.parse_args()

    df = fetch_openfoodfacts(limit=args.limit,
                             ingredient=args.ingr,
                             query=args.query,
                             page_size=args.page_size,
                             all_pages=args.all_pages,
                             nocache=args.nocache)
    if df.empty:
        print("No products found.")
    else:
        print(df.to_string(index=False))

    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()