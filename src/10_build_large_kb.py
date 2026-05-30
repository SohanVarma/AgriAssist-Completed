from pathlib import Path
import pandas as pd
import json

RAW_DATASET = Path("data/raw_farmer_dataset.csv")
OUTPUT_PATH = Path("data/advisory_knowledge_base_large.json")


def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().replace("\n", " ")


def main():
    df = pd.read_csv(RAW_DATASET)

    print("Columns found:")
    print(df.columns)

    records = []

    for idx, row in df.iterrows():
        farmer_message = clean_text(row.get("Farmer_Message", ""))
        advisor_reply = clean_text(row.get("Advisor_Reply", ""))
        crop = clean_text(row.get("Crop", "general"))
        problem_type = clean_text(row.get("Problem_Type", "general"))
        language = clean_text(row.get("Language", "unknown"))
        location = clean_text(row.get("Location", "India"))
        season = clean_text(row.get("Season", "unknown"))
        urgency = clean_text(row.get("Urgency_Level", "unknown"))
        sentiment = clean_text(row.get("Sentiment", "unknown"))
        advisor_type = clean_text(row.get("Advisor_Type", "unknown"))
        product = clean_text(row.get("Product_Recommended", ""))
        intent = clean_text(row.get("Intent_Tag", ""))
        timestamp = clean_text(row.get("Timestamp", ""))

        content = " ".join([
            f"Crop: {crop}.",
            f"Problem Type: {problem_type}.",
            f"Farmer Message: {farmer_message}",
            f"Advisor Reply: {advisor_reply}",
            f"Recommended Product: {product}.",
            f"Location: {location}.",
            f"Season: {season}.",
            f"Urgency: {urgency}.",
            f"Sentiment: {sentiment}.",
            f"Advisor Type: {advisor_type}.",
            f"Intent: {intent}.",
            f"Timestamp: {timestamp}."
        ])

        if len(content.strip()) < 40:
            continue

        records.append({
            "id": f"entry_{idx}",
            "crop": crop,
            "problem_type": problem_type,
            "region": location,
            "season": season,
            "language": language,
            "urgency": urgency,
            "product_recommended": product,
            "intent": intent,
            "content": content
        })

    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Created KB with {len(records)} entries")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
