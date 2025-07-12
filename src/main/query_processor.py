import os
import csv
import logging
import datetime
import warnings

import pandas as pd
import pyterrier as pt
import nltk
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

#from constants.paths import INDEX_OUTPUT_DIR as INDEX_DIR, TEST_QUERIES_PATH, TRAIN_QRELS_PATH, TRAIN_QUERIES_PATH

INDEX_DIR = './indexes'

TRAIN_QUERIES_PATH = 'data/train_queries.csv'
TEST_QUERIES_PATH = 'data/test_queries.csv'
TRAIN_QRELS_PATH = 'data/train_qrels.csv'

# ---------------------- CONFIGURAÇÃO INICIAL ---------------------- #

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
pt.init()

sbert_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Caminho de output
now_str = datetime.datetime.now().strftime("%m_%d_%H.%M")
SUBMISSION_OUTPUT_PATH = f'out/submission_{now_str}.csv'
HITS_PER_QUERY = 5000

# ---------------------- FUNÇÕES AUXILIARES ---------------------- #

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([STEMMER.stem(t) for t in tokens if t.isalnum() and t not in STOPWORDS])

def build_faiss_index(vocab_terms):
    term_embeddings = sbert_model.encode(vocab_terms, convert_to_numpy=True)
    faiss.normalize_L2(term_embeddings)
    index_dim = term_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(index_dim)
    faiss_index.add(term_embeddings)
    return faiss_index, vocab_terms, term_embeddings

def expand_query_sbert(query, retriever, index, top_k_docs=5, top_k_terms=20):
    results = retriever.transform(pd.DataFrame([{"qid": "0", "query": query}]))
    top_docnos = results["docno"].head(top_k_docs).tolist()

    all_terms = set()
    meta = index.getMetaIndex()
    for docno in top_docnos:
        try:
            body = meta.getItem("body", meta.getDocument("docno", docno))
            terms = preprocess_text(body).split()
            all_terms.update(terms)
        except Exception as e:
            logging.warning(f"Erro ao recuperar docno={docno}: {e}")

    terms_list = sorted(all_terms)
    if not terms_list:
        return query

    faiss_index, terms_list, _ = build_faiss_index(terms_list)

    query_emb = sbert_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    _, indices = faiss_index.search(query_emb, top_k_terms)
    expansion_terms = [terms_list[i] for i in indices[0]]

    return query + " " + " ".join(expansion_terms)

# ---------------------- PIPELINE PRINCIPAL ---------------------- #

def run_ltr_with_sbert_expansion():
    index = pt.IndexFactory.of(INDEX_DIR)

    # Recuperadores base
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=HITS_PER_QUERY,
        controls={"bm25.b": 0.25, "bm25.k_1": 1.5, "bm25.k_3": 0.5,
                 'w.0': 2.0, 'w.1': 3.0, 'w.2': 1.25,
                 "qe": "on", "qemodel": "Bo1"},
        metadata=["docno", "body"],
        verbose=True,
        properties={"termpipelines": "Stopwords,PorterStemmer"})

    tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=HITS_PER_QUERY)
    pl2 = pt.terrier.Retriever(index, wmodel="PL2", num_results=HITS_PER_QUERY)
    dfr_bm25 = pt.terrier.Retriever(index, wmodel="DFR_BM25", num_results=HITS_PER_QUERY)
    dph = pt.terrier.Retriever(index, wmodel="DPH", num_results=HITS_PER_QUERY)
    inl2 = pt.terrier.Retriever(index, wmodel="InL2", num_results=HITS_PER_QUERY)
    lgd = pt.terrier.Retriever(index, wmodel="LGD", num_results=HITS_PER_QUERY)
    dfiz = pt.terrier.Retriever(index, wmodel="DFIZ", num_results=HITS_PER_QUERY)

    # ---------------------- DADOS DE TREINAMENTO ---------------------- #

    queries_df = pd.read_csv(TRAIN_QUERIES_PATH)
    queries_df["Query"] = queries_df["Query"].apply(preprocess_text)
    queries_df = queries_df.rename(columns={"QueryId": "qid", "Query": "query"})
    queries_df["qid"] = queries_df["qid"].astype(str)
    queries_df["query_expanded"] = queries_df["query"].apply(
        lambda q: expand_query_sbert(q, bm25, index))

    qrels = pd.read_csv(TRAIN_QRELS_PATH)
    qrels = qrels.rename(columns={"QueryId": "qid", "EntityId": "docno", "Relevance": "label"})
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)

    train_queries, val_queries = train_test_split(queries_df, test_size=0.25, random_state=42)
    qrels_train = qrels[qrels["qid"].isin(train_queries["qid"])]
    qrels_val = qrels[qrels["qid"].isin(val_queries["qid"])]

    # ---------------------- FEATURES E PONTUAÇÃO ---------------------- #

    weights = {
        "tfidf": 1.5,
        "pl2": 1.0,
        "bm25": 1.5,
        "dfr_bm25": 1.0,
        "dph": 1.0,
        "inl2": 1.0,
        "lgd": 1.2,
        "dfiz": 1.5
    }
    
    # Aplicando os pesos usando pt.apply.doc_score
    weighted_tfidf = tfidf >> pt.apply.doc_score(
        lambda r: r["score"] * weights["tfidf"])
    weighted_pl2 = pl2 >> pt.apply.doc_score(
        lambda r: r["score"] * weights["pl2"])
    weighted_bm25 = bm25 >> pt.apply.doc_score(
        lambda r: r["score"] * weights["bm25"])
    weighted_dfr_bm25 = dfr_bm25 >> pt.apply.doc_score(
        lambda r: r["score"] * weights["dfr_bm25"])
    weighted_dph = dph >> pt.apply.doc_score(
        lambda r: r["score"] * weights["dph"])
    weighted_inl2 = inl2 >> pt.apply.doc_score(
        lambda r: r["score"] * weights["inl2"])
    weighted_lgd = lgd >> pt.apply.doc_score(
        lambda r: r["score"] * weights["lgd"])
    weighted_dfiz = dfiz >> pt.apply.doc_score(
        lambda r: r["score"] * weights["dfiz"])
    
    feature_pipeline = bm25 >> (
        weighted_tfidf ** weighted_pl2 ** weighted_bm25 ** weighted_dfr_bm25 ** weighted_dph ** weighted_inl2 ** weighted_lgd ** weighted_dfiz
    )

    # ---------------------- MODELO DE RANQUEAMENTO ---------------------- #

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
        n_jobs=6, 
        eval_at=[10, 20, 100]
    )

    pipeline = feature_pipeline >> pt.ltr.apply_learned_model(ranker, form="ltr")

    logging.info("Treinando modelo com queries expandidas via SBERT...")
    pipeline.fit(train_queries.rename(columns={"query_expanded": "query"}), qrels_train,
                 val_queries.rename(columns={"query_expanded": "query"}), qrels_val)

    val_results = pipeline.transform(val_queries.rename(columns={"query_expanded": "query"}))
    eval_metrics = pt.Evaluate(val_results, qrels_val, metrics=["map", "ndcg", "P_10"])
    for metric, value in eval_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # ---------------------- SUBMISSÃO ---------------------- #

    test_df = pd.read_csv(TEST_QUERIES_PATH)
    test_df["Query"] = test_df["Query"].apply(preprocess_text)
    test_df = test_df.rename(columns={"QueryId": "qid", "Query": "query"})
    test_df["qid"] = test_df["qid"].astype(str)
    test_df["query_expanded"] = test_df["query"].apply(lambda q: expand_query_sbert(q, bm25, index))

    test_results = pipeline.transform(test_df.rename(columns={"query_expanded": "query"}))
    test_results = test_results.groupby("qid").head(100)

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
    with open(SUBMISSION_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(test_results.iterrows(), total=len(test_results)):
            writer.writerow([str(row["qid"]).zfill(3), str(row["docno"]).zfill(7)])

    logging.info(f"Submissão salva em {SUBMISSION_OUTPUT_PATH}")
    return test_results

# ---------------------- EXECUÇÃO ---------------------- #

if __name__ == '__main__':
    run_ltr_with_sbert_expansion()
