import json
import logging
import time
import nltk
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count

nltk.download('punkt')
nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_CORPUS_PATH = './data/corpus.jsonl'
INDEX_DIR = './indexes'
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [STEMMER.stem(t) for t in tokens if t.isalnum() and t not in STOPWORDS]
    return ' '.join(filtered)

def process_line(line):
    if not line.strip():
        return None
    try:
        original_doc = json.loads(line)
        data = original_doc if 'id' in original_doc else original_doc.get('root', {})
        doc_id = data.get('id')
        if not doc_id:
            return None
        title = data.get('title', '')
        text = data.get('text', '')
        keywords = data.get('keywords', [])
        return {
            'docno': doc_id,
            'title': preprocess_text(title),
            'body': preprocess_text(text),
            'keywords': preprocess_text(' '.join(keywords))
        }
    except json.JSONDecodeError:
        return None

def preprocess_corpus_to_df():
    start = time.time()
    logging.info("Iniciando leitura e pré-processamento do corpus...")

    with open(INPUT_CORPUS_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_line, lines, chunksize=100), total=len(lines)))

    docs = [doc for doc in results if doc is not None]
    df = pd.DataFrame(docs)
    logging.info(f"{len(df)} documentos pré-processados em {time.time() - start:.2f} segundos.")
    return df

def create_index(df):
    logging.info("Iniciando criação do índice...")
    start = time.time()
    indexer = pt.terrier.IterDictIndexer(INDEX_DIR, overwrite=True, text_attrs=['title', 'body', 'keywords'], fields=True,  meta=['docno', 'title', 'body', 'keywords'])
    indexref = indexer.index(df.to_dict(orient='records'))
    logging.info(f"Index criado em {time.time() - start:.2f} segundos. Caminho: {INDEX_DIR}")
    return indexref

if __name__ == '__main__':
    start_total = time.time()
    df = preprocess_corpus_to_df()
    create_index(df)
    logging.info(f"Execução total finalizada em {time.time() - start_total:.2f} segundos.")