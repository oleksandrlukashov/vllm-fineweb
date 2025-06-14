from annotator import Annotator
import json
import duckdb
import pandas as pd
from tqdm import tqdm

class FineWebMultiLang:
    def __init__(self, languages, download_limit: int = 100_000):
        self.languages = languages
        self.download_limit = download_limit
        self.dataset = pd.DataFrame(columns=['text', 'lang'])

    def save_to_json(self, path: str):
        self.dataset.to_json(path, orient='records', lines=True, force_ascii=False)

    def load_datasets(self):
        for lang in tqdm(self.languages):
            print('Loading subset:', lang)
            if lang == 'CC-MAIN-2023-40':
                query = f"""
                SELECT text
                FROM 'hf://datasets/HuggingFaceFW/fineweb/data/{lang}/000_00000.parquet'
                LIMIT {self.download_limit};
                """
                current_lang = 'en'
            else:
                query = f"""
                SELECT text
                FROM 'hf://datasets/HuggingFaceFW/fineweb-2/data/{lang}/train/000_00000.parquet'
                LIMIT {self.download_limit};
                """
                current_lang = lang
            df = duckdb.query(query).to_df()
            df['lang'] = current_lang
            df.to_json(f'raw_data/euro_additional/{current_lang}.json', orient='records', indent=2, force_ascii=False)
            self.dataset = pd.concat([self.dataset, df], ignore_index=True)
        return self.dataset.to_dict(orient='records')
    
euro_main = ['deu_Latn', 'spa_Latn', 'fra_Latn', 'ita_Latn']
euro = FineWebMultiLang(languages=euro_main, download_limit=5_000)
euro = euro.load_datasets()
annotator = Annotator(llm='Qwen/Qwen3-4B', quantization=None, batch_size=32, temperature=0.5)
annotator.run(dataset=euro, save_path='euro_main_synth.jsonl')
