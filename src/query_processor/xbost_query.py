import os
import csv
import logging
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import xgboost as xgb
from sklearn.model_selection import train_test_split

from constants.paths import INDEX_OUTPUT_DIR, TEST_QUERIES_PATH, TRAIN_QRELS_PATH, TRAIN_QUERIES_PATH

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

pt.java.init()


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [STEMMER.stem(t)
                for t in tokens if t.isalnum() and t not in STOPWORDS]
    return ' '.join(filtered)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

now_str = datetime.datetime.now().strftime("%m_%d_%H.%M")
SUBMISSION_OUTPUT_PATH = f'out/submission_{now_str}.csv'

HITS_PER_QUERY = 200

def run_ltr_pipeline():
    index = pt.IndexFactory.of(INDEX_OUTPUT_DIR)

    train_queries = pd.read_csv(TRAIN_QUERIES_PATH)
    train_queries["Query"] = train_queries["Query"].apply(preprocess_text)
    train_queries = train_queries.rename(
        columns={"QueryId": "qid", "Query": "query"})
    train_queries["qid"] = train_queries["qid"].astype(str)

    qrels = pd.read_csv(TRAIN_QRELS_PATH)
    qrels = qrels.rename(
        columns={"QueryId": "qid", "EntityId": "docno", "Relevance": "label"})
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)

    train_queries_train, train_queries_val = train_test_split(
        train_queries, test_size=0.2, random_state=42)
    train_qrels_train = qrels[qrels['qid'].isin(train_queries_train['qid'])]
    train_qrels_val = qrels[qrels['qid'].isin(train_queries_val['qid'])]

    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=HITS_PER_QUERY,
                                controls={"bm25.b": 0.25, "bm25.k_1": 1.5, "bm25.k_3": 0.5,
                                          'w.0': 1, 'w.1': 2, 'w.2': 1},
                                metadata=["docno", "body"], verbose=True)

    tfidf = pt.terrier.Retriever(
        index, wmodel="TF_IDF", num_results=HITS_PER_QUERY)
    pl2 = pt.terrier.Retriever(index, wmodel="PL2", num_results=HITS_PER_QUERY)

    rerank_feats = tfidf ** pl2 ** bm25
    ltr_feats_pipeline = bm25 >> rerank_feats

    print("Pipeline de recursos LTR:")
    print(ltr_feats_pipeline)

    logging.info("Extraindo features para treinamento com XGBoost...")

    # Etapa importante: associar os rótulos (label) aos resultados com `.fit()` do PyTerrier
    ltr_feats_pipeline.fit(train_queries_train, train_qrels_train)
    X_train = ltr_feats_pipeline.transform(train_queries_train)

    ltr_feats_pipeline.fit(train_queries_val, train_qrels_val)
    X_val = ltr_feats_pipeline.transform(train_queries_val)

    X_train = X_train.sort_values(["qid", "docno"])
    X_val = X_val.sort_values(["qid", "docno"])

    X_train_feat = X_train.drop(columns=["qid", "docno", "label"])
    y_train = X_train["label"]
    group_train = X_train.groupby("qid").size().to_numpy()

    X_val_feat = X_val.drop(columns=["qid", "docno", "label"])
    y_val = X_val["label"]
    group_val = X_val.groupby("qid").size().to_numpy()

    xgb_model = xgb.XGBRanker(
        objective='rank:ndcg',
        eval_metric='ndcg',
        booster='gbtree',
        max_depth=7,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1
    )

    logging.info("Treinando XGBoostRanker com early stopping...")
    xgb_model.fit(
        X_train_feat, y_train, group=group_train,
        eval_set=[(X_val_feat, y_val)],
        eval_group=[group_val],
        early_stopping_rounds=50,
        verbose=True
    )

    # Aplicar modelo XGBoost como reranker
    def xgb_predict(df):
        features = df.drop(columns=["qid", "docno"])
        df = df.copy()
        df["score"] = xgb_model.predict(features)
        return df

    ltr_pipeline = ltr_feats_pipeline >> pt.apply.doc_model(xgb_predict)

    logging.info("Avaliando desempenho no conjunto de treino...")
    results = ltr_pipeline.transform(train_queries)
    eval_metrics = pt.Utils.evaluate(results, qrels, metrics=[
                                     "map", "ndcg", "recall", "P.10"])
    for metric, value in eval_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    test_queries = pd.read_csv(TEST_QUERIES_PATH)
    test_queries["Query"] = test_queries["Query"].apply(preprocess_text)
    test_queries = test_queries.rename(
        columns={"QueryId": "qid", "Query": "query"})
    test_queries["qid"] = test_queries["qid"].astype(str)

    logging.info("Executando pipeline no conjunto de teste...")
    test_results = ltr_pipeline.transform(test_queries)
    test_results = test_results.groupby("qid").head(100)

    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
    with open(SUBMISSION_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(test_results.iterrows(), total=len(test_results), desc="Gerando submissão"):
            writer.writerow([str(row["qid"]).zfill(3),
                            str(row["docno"]).zfill(7)])

    logging.info(f"Submissão salva em {SUBMISSION_OUTPUT_PATH}")


if __name__ == '__main__':
    run_ltr_pipeline()
