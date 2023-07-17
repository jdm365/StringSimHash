import pandas as pd
from transformers import pipeline
from tqdm import tqdm


# Function to perform translation
def translate(text, translator_a, translator_b):
    translation = translator_a(
            text, 
            max_length=int(len(text) * 1.5), 
            num_beams=2, 
            early_stopping=True
            )
    translation = translation[0]['translation_text']

    back_translation = translator_b(
            translation, 
            max_length=int(len(text) * 1.4), 
            num_beams=2, 
            early_stopping=True
            )
    back_translation = back_translation[0]['translation_text']
    return back_translation 


if __name__ == "__main__":
    MODEL_NAME_A = 'Helsinki-NLP/opus-mt-en-fr'
    MODEL_NAME_B = 'Helsinki-NLP/opus-mt-fr-en'

    translator_a = pipeline("translation", model=MODEL_NAME_A, tokenizer=MODEL_NAME_A, device=0)
    translator_b = pipeline("translation", model=MODEL_NAME_B, tokenizer=MODEL_NAME_B, device=0)

    text = "TEA MALDAZE"
    df = pd.read_feather('corrupted_companies_dedup.feather')

    df = df.sample(100)

    tqdm.pandas()
    df['corrpted_name'] = df['company'].progress_apply(lambda x: translate(x, translator_a, translator_b))
    print(df.head(25))
