import os
import csv
import logging
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import datetime
import nltk
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings

from neural_reranking.monot5 import monoT5

from constants.paths import TRAIN_QUERIES_PATH,TEST_QUERIES_PATH,TRAIN_QRELS_PATH,EXPANDED_INDEX_DIR as INDEX_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
pt.java.init()
sbert = SentenceTransformer(
    'multi-qa-mpnet-base-dot-v1')  # Troca do modelo SBERT

# Configuração de caminhos
now_str = datetime.datetime.now().strftime("%m_%d_%H.%M")

SUBMISSION_OUTPUT_PATH = f'out/submission_{now_str}.csv'


HITS_PER_QUERY = 200

# --- FUNÇÕES AUXILIARES ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([STEMMER.stem(t) for t in tokens if t.isalnum() and t not in STOPWORDS])


def build_faiss_index(vocab_terms):
    term_embeddings = sbert.encode(vocab_terms, convert_to_numpy=True)
    faiss.normalize_L2(term_embeddings)
    dim = term_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(term_embeddings)
    return index, vocab_terms, term_embeddings


def expand_query_sbert(query, retriever, index, top_k_docs=5, top_k_terms=20):
    # Recupera os documentos mais relevantes da query original
    hits = retriever.transform(pd.DataFrame([{"qid": "0", "query": query}]))
    top_docs = hits["docno"].head(top_k_docs).tolist()

    # Extrai termos dos corpos dos documentos
    all_terms = set()
    meta = index.getMetaIndex()
    for docno in top_docs:
        try:
            body = meta.getItem("body", meta.getDocument("docno", docno))
            terms = preprocess_text(body).split()
            all_terms.update(terms)
        except Exception as e:
            logging.warning(f"Erro ao recuperar docno={docno}: {e}")

    # Indexa termos para busca vetorial
    terms_list = sorted(all_terms)
    if not terms_list:
        return query  # fallback

    faiss_index, terms_list, _ = build_faiss_index(terms_list)

    query_emb = sbert.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    _, indices = faiss_index.search(query_emb, top_k_terms)
    expansion_terms = [terms_list[i] for i in indices[0]]

    return query + " " + " ".join(expansion_terms)


# --- PIPELINE PRINCIPAL ---
def run_ltr_with_sbert_expansion(model_type='lgb'):
    assert model_type in ['lgb', 'xgb']
    index = pt.IndexFactory.of(INDEX_DIR)

    controls = {
        'w.0': 1.5,  # title_basic
        'w.1': 2.5,  # body_basic
        'w.2': 2.0,  # title_advanced
        'w.3': 3.0,  # body_advanced
        'w.4': 1.2,  # title_ngrams
        'w.5': 1.8,  # body_ngrams
        'w.6': 1.0,  # keywords_basic
        'w.7': 1.5,  # keywords_advanced
        'w.8': 2.5   # keyphrases
    }

    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=HITS_PER_QUERY,
                                controls={"bm25.b": 0.25, "bm25.k_1": 1.5, "bm25.k_3": 0.5,
                                          'w.0': 1.5,  # title_basic
                                          'w.1': 2.5,
                                          'w.2': 2.0,
                                          'w.3': 3.0,
                                          'w.4': 1.2,
                                          'w.5': 1.8,
                                          'w.6': 1.0,
                                          'w.7': 1.5,
                                          'w.8': 2.5,
                                          "qe": "on", "qemodel": "Bo1"},
                                metadata=["docno", "body"],
                                verbose=True,
                                properties={
                                    "termpipelines": "Stopwords,PorterStemmer"}

                                )

    tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=HITS_PER_QUERY,
                                 controls=controls, metadata=["docno", "body"])
    pl2 = pt.terrier.Retriever(index, wmodel="PL2", num_results=HITS_PER_QUERY,
                               controls=controls, metadata=["docno", "body"])

    dfr_bm25 = pt.terrier.Retriever(
        index, wmodel="DFR_BM25", num_results=HITS_PER_QUERY, controls=controls)
    dph = pt.terrier.Retriever(
        index, wmodel="DPH", num_results=HITS_PER_QUERY, controls=controls)

    inl2 = pt.terrier.Retriever(
        index, wmodel="InL2", num_results=HITS_PER_QUERY)
    lgd = pt.terrier.Retriever(
        index, wmodel="LGD", num_results=HITS_PER_QUERY, controls=controls)
    dfiz = pt.terrier.Retriever(
        index, wmodel="DFIZ", num_results=HITS_PER_QUERY, controls=controls)

    train_queries = pd.read_csv(TRAIN_QUERIES_PATH)
    train_queries["Query"] = train_queries["Query"].apply(preprocess_text)
    train_queries = train_queries.rename(
        columns={"QueryId": "qid", "Query": "query"})
    train_queries["qid"] = train_queries["qid"].astype(str)

    all_terms = set()
    for q in train_queries["query"]:
        all_terms.update(q.split())
    vocab_terms = sorted(all_terms)
    faiss_index, vocab_terms, _ = build_faiss_index(vocab_terms)

    train_queries["query_expanded"] = train_queries["query"].apply(
        lambda q: expand_query_sbert(
            q, bm25, index, top_k_docs=5, top_k_terms=20)
    )

    qrels = pd.read_csv(TRAIN_QRELS_PATH)
    qrels = qrels.rename(
        columns={"QueryId": "qid", "EntityId": "docno", "Relevance": "label"})
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)

    train_q, val_q = train_test_split(
        train_queries, test_size=0.2, random_state=42)
    qrels_train = qrels[qrels["qid"].isin(train_q["qid"])]
    qrels_val = qrels[qrels["qid"].isin(val_q["qid"])]

    feature_pipe = bm25 >> monoT5 >> (
        tfidf ** pl2 ** bm25 ** dfr_bm25**dph**inl2**lgd**dfiz)

    ranker = lgb.LGBMRanker(
        task="train",
        min_data_in_leaf=10,
        min_sum_hessian_in_leaf=1,
        max_bin=511,
        num_leaves=31,
        max_depth=20,
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.1,
        importance_type="gain",
        num_iterations=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=6
    )

    pipeline = feature_pipe >> pt.ltr.apply_learned_model(ranker, form="ltr")

    logging.info("Treinando modelo com queries expandidas via SBERT...")
    pipeline.fit(train_q.rename(columns={"query_expanded": "query"}), qrels_train,
                 val_q.rename(columns={"query_expanded": "query"}), qrels_val)

    val_results = pipeline.transform(
        val_q.rename(columns={"query_expanded": "query"}))
    metrics = pt.Evaluate(val_results, qrels_val,
                          metrics=["map", "ndcg", "P_10"])
    for m, v in metrics.items():
        logging.info(f"{m}: {v:.4f}")

    test_queries = pd.read_csv(TEST_QUERIES_PATH)
    test_queries["Query"] = test_queries["Query"].apply(preprocess_text)
    test_queries = test_queries.rename(
        columns={"QueryId": "qid", "Query": "query"})
    test_queries["qid"] = test_queries["qid"].astype(str)
    test_queries["query_expanded"] = test_queries["query"].apply(
        lambda q: expand_query_sbert(
            q, bm25, index, top_k_docs=5, top_k_terms=20)
    )

    test_results = pipeline.transform(
        test_queries.rename(columns={"query_expanded": "query"}))
    test_results = test_results.groupby("qid").head(100)

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
    with open(SUBMISSION_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(test_results.iterrows(), total=len(test_results)):
            writer.writerow([str(row["qid"]).zfill(3),
                            str(row["docno"]).zfill(7)])

    logging.info(f"Submissão salva em {SUBMISSION_OUTPUT_PATH}")
    return test_results


if __name__ == '__main__':
    run_ltr_with_sbert_expansion('lgb')
