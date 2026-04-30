from pathlib import Path
RESULTS_DIR=Path('results'); RESULTS_DIR.mkdir(exist_ok=True)
KB={'leaf_spot':'Remove heavily infected leaves, avoid overhead irrigation, improve spacing, and consult a local agriculture officer for recommended fungicide usage.'}
def main():
    disease='leaf_spot'; confidence=0.82; region='India'; advice=KB[disease]
    text=f'''# Sample Localized Crop Advisory\n\n## Predicted Disease\n{disease}\n\n## Confidence\n{confidence:.2f}\n\n## Region\n{region}\n\n## Retrieved Advisory\n{advice}\n\n## Ethics and Safety Note\nThis is AI-assisted guidance. Farmers should confirm pesticide or chemical use with local agricultural experts before applying treatment.\n'''
    (RESULTS_DIR/'sample_advisory.md').write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__': main()
