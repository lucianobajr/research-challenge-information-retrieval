import os
import json
import time
import logging
import nltk
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count

from constants.paths import CORPUS_PATH, LEMMATIZER_INDEX_PATH

# Downloads necessários do NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class LemmatizerIndexer:
    def __init__(self, corpus_path: str, index_dir: str):
        self.corpus_path = corpus_path
        self.index_dir = index_dir

        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        if not pt.started():
            pt.init()

    def _lemmatize_or_stem(self, token: str) -> str:
        """Aplica lematização, e fallback com stemming"""
        try:
            lemma = self.lemmatizer.lemmatize(token)
            return lemma if lemma else self.stemmer.stem(token)
        except:
            return self.stemmer.stem(token)

    def _preprocess_text(self, text: str) -> str:
        """Pré-processamento com lematização + stemming e remoção de stopwords"""
        tokens = word_tokenize(text.lower())
        processed = [self._lemmatize_or_stem(
            t) for t in tokens if t.isalnum() and t not in self.stopwords]
        return ' '.join(processed)

    def _process_document(self, line: str) -> dict | None:
        if not line.strip():
            return None
        try:
            raw = json.loads(line)
            doc = raw if 'id' in raw else raw.get('root', {})
            doc_id = doc.get('id')
            if not doc_id:
                return None

            title = doc.get('title', '')
            body = doc.get('text', '')
            keywords = doc.get('keywords', [])

            return {
                'docno': doc_id,
                'title': self._preprocess_text(title),
                'body': self._preprocess_text(body),
                'keywords': self._preprocess_text(' '.join(keywords))
            }
        except json.JSONDecodeError:
            return None

    def preprocess_corpus(self) -> pd.DataFrame:
        """Lê e pré-processa o corpus, retornando um DataFrame"""
        logging.info("Iniciando leitura e pré-processamento do corpus...")
        start = time.time()

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with Pool(processes=cpu_count()) as pool:
            processed = list(tqdm(pool.imap_unordered(
                self._process_document, lines, chunksize=100), total=len(lines)))

        docs = [doc for doc in processed if doc is not None]
        df = pd.DataFrame(docs)
        logging.info(
            f"{len(df)} documentos pré-processados em {time.time() - start:.2f} segundos.")
        return df

    def build_index(self, df: pd.DataFrame):
        """Cria o índice PyTerrier com textos lematizados"""
        logging.info("Iniciando criação do índice...")
        start = time.time()

        indexer = pt.IterDictIndexer(
            self.index_dir,
            overwrite=True,
            text_attrs=['title', 'body', 'keywords'],
            fields=True,
            meta=['docno', 'title', 'body', 'keywords']
        )

        index_ref = indexer.index(df.to_dict(orient='records'))
        logging.info(
            f"Índice criado em {time.time() - start:.2f} segundos. Caminho: {self.index_dir}")
        return index_ref


if __name__ == '__main__':

    lemmatizer_indexer = LemmatizerIndexer(CORPUS_PATH, LEMMATIZER_INDEX_PATH)

    start_total = time.time()
    df = lemmatizer_indexer.preprocess_corpus()
    index_ref = lemmatizer_indexer.build_index(df)
    logging.info(
        f"Execução completa finalizada em {time.time() - start_total:.2f} segundos.")
