from pathlib import Path
import pandas as pd
import json

RAW_DATASET = Path('data/raw_farmer_dataset.csv')
OUTPUT_PATH = Path('data/advisory_knowledge_base_large.json')


def clean_text(text):
    if pd.isna(text):
        return ''

    return str(text).strip().replace('\n', ' ')


def main():
    df = pd.read_csv(RAW_DATASET)

    print(df.columns)

    records = []

    for idx, row in df.iterrows():

        combined_text = ' '.join([
            clean_text(str(v))
            for v in row.values
        ])

        if len(combined_text) < 40:
            continue

        records.append({
            'id': f'entry_{idx}',
            'disease': clean_text(row.iloc[0]),
            'crop': 'general',
            'region': 'India',
            'language': 'english',
            'content': combined_text
        })

    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2),
        encoding='utf-8'
    )

    print(f'Created KB with {len(records)} entries')
    print(f'Saved to: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()