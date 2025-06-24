import os
import json
import time
import logging
import nltk
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count

from constants.paths import CORPUS_PATH, INDEX_OUTPUT_DIR

# Downloads iniciais
nltk.download('punkt')
nltk.download('stopwords')

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class CorpusIndexer:
    def __init__(self, corpus_path: str, index_dir: str):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

        if not pt.started():
            pt.init()

    def _preprocess_text(self, text: str) -> str:
        """Tokeniza, remove stopwords e aplica stemming"""
        tokens = word_tokenize(text.lower())
        filtered = [self.stemmer.stem(token)
                    for token in tokens if token.isalnum() and token not in self.stopwords]
        return ' '.join(filtered)

    def _process_json_line(self, line: str) -> dict | None:
        """Processa uma linha do corpus JSONL"""
        if not line.strip():
            return None
        try:
            record = json.loads(line)
            doc = record if 'id' in record else record.get('root', {})
            doc_id = doc.get('id')
            if not doc_id:
                return None

            title = doc.get('title', '')
            text = doc.get('text', '')
            keywords = doc.get('keywords', [])

            return {
                'docno': doc_id,
                'title': self._preprocess_text(title),
                'body': self._preprocess_text(text),
                'keywords': self._preprocess_text(' '.join(keywords))
            }
        except json.JSONDecodeError:
            return None

    def preprocess_corpus(self) -> pd.DataFrame:
        """Lê e pré-processa o corpus retornando um DataFrame"""
        logging.info("Lendo e pré-processando o corpus...")
        start = time.time()

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with Pool(processes=cpu_count()) as pool:
            processed = list(tqdm(pool.imap_unordered(
                self._process_json_line, lines, chunksize=100), total=len(lines)))

        docs = [doc for doc in processed if doc is not None]
        df = pd.DataFrame(docs)
        logging.info(
            f"{len(df)} documentos processados em {time.time() - start:.2f} segundos.")
        return df

    def build_index(self, dataframe: pd.DataFrame):
        """Cria o índice PyTerrier a partir do DataFrame processado"""
        logging.info("Criando índice PyTerrier...")
        start = time.time()

        indexer = pt.IterDictIndexer(
            self.index_dir,
            overwrite=True,
            fields=True,
            text_attrs=['title', 'body', 'keywords'],
            meta=['docno', 'title', 'body', 'keywords']
        )

        index_ref = indexer.index(dataframe.to_dict(orient='records'))
        logging.info(
            f"Índice criado em {time.time() - start:.2f} segundos em {self.index_dir}")
        return index_ref


if __name__ == '__main__':
    indexer = CorpusIndexer(CORPUS_PATH, INDEX_OUTPUT_DIR)

    start_time = time.time()

    # Etapas
    df_corpus = indexer.preprocess_corpus()
    index_ref = indexer.build_index(df_corpus)

    logging.info(
        f"Pipeline completo em {time.time() - start_time:.2f} segundos.")
