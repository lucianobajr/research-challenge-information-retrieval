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
from nltk.tokenize import word_tokenize, sent_tokenize
from multiprocessing import Pool, cpu_count
from gensim.models.phrases import Phrases, Phraser

from constants.paths import EXPANDED_INDEX_DIR, EXPANDED_INDEX_PATH, CORPUS_PATH


# Downloads necessários
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ExpandedIndexer:
    def __init__(self, corpus_path: str, index_dir_basic: str, index_dir_expanded: str):
        self.corpus_path = corpus_path
        self.index_dir_basic = index_dir_basic
        self.index_dir_expanded = index_dir_expanded

        # Stopwords customizadas
        custom_stopwords = {
            "the", "and", "to", "of", "a", "in", "that", "is", "was", "for", "on", "it", "with",
            "as", "are", "at", "be", "this", "by", "have", "from", "or", "an", "but", "not", "what",
            "all", "were", "when", "we", "there", "can", "been", "who", "will", "no", "more", "if",
            "out", "so", "up", "said", "about", "than", "its", "into", "them", "only", "just"
        }
        self.stopwords = set(stopwords.words('english')
                             ).union(custom_stopwords)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        if not pt.started():
            pt.init()

    def _preprocess_basic(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        filtered = [self.stemmer.stem(
            t) for t in tokens if t.isalnum() and t not in self.stopwords]
        return ' '.join(filtered)

    def _preprocess_advanced(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        filtered = [self.lemmatizer.lemmatize(
            t) for t in tokens if t.isalnum() and t not in self.stopwords]
        return ' '.join(filtered)

    def _generate_ngrams(self, text: str, n_range=(2, 3)) -> str:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            t for t in tokens if t.isalnum() and t not in self.stopwords]

        ngrams = []
        for n in range(n_range[0], n_range[1] + 1):
            if len(filtered_tokens) < n:
                continue
            ngrams += ['_'.join(filtered_tokens[i:i+n])
                       for i in range(len(filtered_tokens)-n+1)]

        return ' '.join(filtered_tokens + ngrams)

    def _extract_keyphrases(self, text: str) -> str:
        sentences = sent_tokenize(text.lower())
        tokenized = [word_tokenize(sent) for sent in sentences]
        filtered = [[w for w in sent if w.isalnum() and w not in self.stopwords]
                    for sent in tokenized]

        bigrams = Phrases(filtered, min_count=2, threshold=10)
        trigrams = Phrases(bigrams[filtered], threshold=10)

        bigram_phraser = Phraser(bigrams)
        trigram_phraser = Phraser(trigrams)

        phrases = []
        for sent in filtered:
            phrases.extend(trigram_phraser[bigram_phraser[sent]])

        return ' '.join(phrases)

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
                'title': title,
                'body': body,
                'title_basic': self._preprocess_basic(title),
                'body_basic': self._preprocess_basic(body),
                'title_advanced': self._preprocess_advanced(title),
                'body_advanced': self._preprocess_advanced(body),
                'title_ngrams': self._generate_ngrams(title),
                'body_ngrams': self._generate_ngrams(body),
                'keywords_basic': self._preprocess_basic(' '.join(keywords)),
                'keywords_advanced': self._preprocess_advanced(' '.join(keywords)),
                'keyphrases': self._extract_keyphrases(body)
            }
        except json.JSONDecodeError:
            return None

    def preprocess_corpus(self) -> pd.DataFrame:
        """Lê e processa o corpus JSONL em DataFrame"""
        logging.info("Lendo e processando o corpus...")
        start = time.time()

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with Pool(cpu_count()) as pool:
            processed = list(tqdm(pool.imap_unordered(
                self._process_document, lines, chunksize=100), total=len(lines)))

        docs = [doc for doc in processed if doc]
        df = pd.DataFrame(docs)
        logging.info(
            f"{len(df)} documentos processados em {time.time() - start:.2f} segundos.")
        return df

    def build_indices(self, df: pd.DataFrame):
        """Cria dois índices: básico e expandido"""
        logging.info("Criando índice básico...")
        basic_indexer = pt.IterDictIndexer(
            self.index_dir_basic,
            overwrite=True,
            text_attrs=['title_basic', 'body_basic', 'keywords_basic'],
            fields=True,
            meta=['docno', 'title', 'body']
        )
        basic_ref = basic_indexer.index(df.to_dict(orient='records'))

        logging.info("Criando índice expandido com técnicas avançadas...")
        expanded_indexer = pt.IterDictIndexer(
            self.index_dir_expanded,
            overwrite=True,
            text_attrs=[
                'title_advanced', 'body_advanced', 'keywords_advanced',
                'title_ngrams', 'body_ngrams', 'keyphrases'
            ],
            fields=True,
            meta=['docno', 'title', 'body']
        )
        expanded_ref = expanded_indexer.index(df.to_dict(orient='records'))

        logging.info("Índices criados com sucesso.")
        return basic_ref, expanded_ref


if __name__ == '__main__':

    indexer = ExpandedIndexer(
        corpus_path=CORPUS_PATH,
        index_dir_basic=EXPANDED_INDEX_PATH,
        index_dir_expanded=EXPANDED_INDEX_DIR
    )

    total_start = time.time()

    df = indexer.preprocess_corpus()
    basic_ref, expanded_ref = indexer.build_indices(df)

    logging.info(
        f"Pipeline finalizado em {time.time() - total_start:.2f} segundos.")
