import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from comp_model.sdk.scrapers.factory import ScraperFactory
from comp_model.sdk.model import CompModelModel
from geronimo.artifacts import ArtifactStore

def main():
    parser = argparse.ArgumentParser(description="Predict real estate comp value from URL or MLS ID.")
    parser.add_argument("--url", type=str, required=False, help="Live Zillow/Realtor URL")
    parser.add_argument("--mls-id", type=str, required=False, help="RESO Web API Property ListingKey")
    args = parser.parse_args()
    
    if not args.url and not args.mls_id:
        print("Error: You must provide either a --url or an --mls-id.")
        sys.exit(1)
        
    features = None
    target_id = ""
    
    if args.url:
        target_id = args.url
        print(f"Resolving scraper for: {args.url}")
        scraper = ScraperFactory.get_scraper(args.url)
        if not scraper:
            print("Error: Unsupported URL domain. Scaffold a new Scraper for this site.")
            sys.exit(1)
            
        print("Scraping property features via public web...")
        features = scraper.extract_raw_features()

        # Enrich with City of Rochester assessor data when the scraper couldn't
        # get assessment fields (Zillow gates these behind a PerimeterX-protected
        # GraphQL call). Only fills fields the scraper left empty.
        if not features.get("assessed_value"):
            from comp_model.sdk.data_sources import lookup_rochester_assessment
            address = features.get("address")
            zip_code = features.get("zip_code")
            print(f"Assessment missing from scrape; querying City of Rochester for '{address}'...")
            enrichment = lookup_rochester_assessment(address, zip_code)
            if enrichment:
                filled = []
                for k, v in enrichment.items():
                    existing = features.get(k)
                    if existing in (None, 0, 0.0, "", "Unknown"):
                        features[k] = v
                        filled.append(k)
                print(f"   Enrichment filled: {filled}")
            else:
                print("   No Rochester record found. (Only City of Rochester properties "
                      "are covered - Monroe County suburbs aren't in this MapServer.)")

        # Sanity check: if enrichment brought in a non-zero second_floor_area,
        # the property has at least 2 stories.
        second_floor = pd.to_numeric(features.get("second_floor_area"), errors="coerce")
        if pd.notna(second_floor) and second_floor > 0:
            features["stories"] = 2
            print("   Sanitized stories = 2 (second_floor_area is non-zero).")

    elif args.mls_id:
        target_id = args.mls_id
        from comp_model.sdk.mls import MLSClient
        print(f"Authenticating MLS Client for Listing: {args.mls_id}...")
        client = MLSClient()
        
        print("Fetching RESO Web API payload...")
        features = client.fetch_property(args.mls_id)
        if not features:
            print("Error: Failed to fetch property from MLS. Check credentials or listing ID.")
            sys.exit(1)

    # Natively import macro features dynamically
    from comp_model.sdk.data_sources import training_macro, UNIFIED_COLUMNS
    import os
    from dotenv import load_dotenv
    # FRED_API_KEY lives in the project root comp_model/.env (three dirs up from experiments/)
    _ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
    load_dotenv(_ENV_PATH)

    # We must construct a DataFrame compliant with the geronimo FeatureSet constraints
    df = pd.DataFrame([features])

    # Backfill any column the scraper/MLS left out so transform() emits the full feature
    # matrix the estimator was trained on (it silently skips columns that are absent).
    # Macro columns are handled after the FRED merge below to avoid cross-join suffixes.
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Needs a sale_date for Macro features and H3 lookups!
    from datetime import datetime
    df['sale_date'] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))

    # Fetch today's macro indicators
    macro_df = training_macro.load()
    if not macro_df.empty:
        # Forward fill up to today since FRED might be delayed 1-2 days
        # macro_df = macro_df.set_index('sale_date').resample('D').ffill().reset_index()
        # df = df.merge(macro_df, on='sale_date', how='left')
        # 1. Get the row with the maximum sale_date
        latest_macro_df = macro_df.loc[[macro_df['sale_date'].idxmax()]]

        # 2. Drop 'sale_date' from the macro slice so it doesn't try to match on it,
        # then merge using a cross join to apply it to every row in df
        latest_macro_df = latest_macro_df.drop(columns=['sale_date'])
        df = df.merge(latest_macro_df, how='cross')

    # Guarantee the macro columns exist even if FRED was unavailable/empty, so the
    # feature matrix stays complete (XGBoost tolerates NaN).
    for col in ["mortgage_30y", "fed_funds", "cpi", "unemployment"]:
        if col not in df.columns:
            df[col] = np.nan

    # Spatially join the School District!
    from comp_model.sdk.data_sources import _assign_school_districts
    df = _assign_school_districts(df)
        
    print("Loading XGBoost Model...")
    store = ArtifactStore(project="comp_model", version="1.0.0")
    model = CompModelModel()
    model.load(store)

    # Demo placeholder: when the assessor lookup couldn't supply an assessment, fall back
    # to half the Zillow asking price (which the scraper stores in 'sale_price').
    assessed = df.at[0, "assessed_value"]
    if pd.isna(assessed) or assessed in (0, 0.0):
        asking = features.get("sale_price") or 0
        print(f"WARNING: assessed_value unavailable; using sale_price/2 = ${asking / 2:,.0f} "
              f"as a demo placeholder (prediction will be degraded).")
        df.at[0, "assessed_value"] = asking / 2.0

    # transform() is what predict() runs internally - it computes the geo-velocity H3
    # lookups, the macro/FRED columns, and the encoded school_district. Print it so the
    # model's actual inputs are visible.
    X = model.features.transform(df)
    print("\nTransformed feature matrix (the model's actual inputs):")
    print(X.T)

    print("\nPredicting Sale Price...")
    pred_price = model.predict(df)[0]
    
    actual_price = features.get("sale_price", 0)
    
    print("\n" + "="*50)
    print(f"PROPERTY PREDICTION: {target_id}")
    print("="*50)
    if actual_price:
        print(f"Listed Price:    ${actual_price:,.2f}")
    print(f"Predicted Value: ${pred_price:,.2f}")
    if actual_price:
        diff = pred_price - actual_price
        pct = (diff / actual_price) * 100
        print(f"Difference:      ${diff:,.2f} ({pct:+.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
