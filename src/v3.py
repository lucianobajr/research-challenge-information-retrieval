import pyterrier as pt
import pandas as pd
import json
import os
import re
from tqdm import tqdm
from pyterrier.terrier import Retriever

from ir_measures import MAP, nDCG, Recall

# --- Configuração ---
CORPUS_FILE = './data/corpus.jsonl'
QUERIES_FILE = './data/test_queries.csv'
TRAIN_QUERIES_FILE = './data/train_queries.csv'
QRELS_FILE = './data/train_qrels.csv'
OUTPUT_FILE = './out/submission.csv'
INDEX_DIR = './terrier_index'
MAX_RESULTS_PER_QUERY = 100

# --- Inicializa PyTerrier ---
print("Inicializando PyTerrier...")
pt.java.add_package("com.github.terrierteam", "terrier-prf", "-SNAPSHOT")
pt.java.init()

# --- Prepara o Corpus ---
def prepare_corpus(path):
    docs = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Lendo corpus"):
            try:
                data = json.loads(line)
                title = data.get('title', '')
                keywords = data.get('keywords', '')
                text = data.get('text', '')
                if isinstance(keywords, list):
                    keywords = ' '.join(keywords)
                combined = f"{title} {keywords} {text}".strip()
                doc_id = data.get('id')
                if doc_id and combined:
                    docs.append({'docno': str(doc_id), 'text': combined})
            except:
                continue
    return pd.DataFrame(docs)

# --- Indexa o Corpus se não existir ---
if not os.path.exists(INDEX_DIR + "/data.properties"):
    corpus_df = prepare_corpus(CORPUS_FILE)
    print(f"{len(corpus_df)} documentos prontos para indexação.")
    indexer = pt.IterDictIndexer(INDEX_DIR, meta={'docno': 20}, overwrite=True)
    index_ref = indexer.index(corpus_df.to_dict(orient='records'))
else:
    index_ref = pt.IndexRef.of(INDEX_DIR + "/data.properties")

index = pt.IndexFactory.of(index_ref)
print(f"Index carregado com {index.getCollectionStatistics().getNumberOfDocuments()} documentos.")

# --- Define modelos ---
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=MAX_RESULTS_PER_QUERY)
dfr = pt.BatchRetrieve(index, wmodel="DFR_BM25", num_results=MAX_RESULTS_PER_QUERY)
pl2 = pt.BatchRetrieve(index, wmodel="PL2", num_results=MAX_RESULTS_PER_QUERY)
qe = pt.rewrite.Bo1QueryExpansion(index)

# --- Combinação dos modelos com expansão de consulta ---
bm25_qe = qe >> bm25
fusion = CombSum([bm25_qe, dfr, pl2])

# --- Carrega e limpa as queries ---
def load_and_clean_queries(file_path):
    df = pd.read_csv(file_path, dtype={'QueryId': str})
    df = df.rename(columns={'QueryId': 'qid', 'Query': 'query'})
    df['qid'] = df['qid'].astype(str)
    df['query'] = df['query'].apply(lambda q: re.sub(r"[^a-zA-Z0-9\s]", " ", str(q)))
    return df

queries_df = load_and_clean_queries(QUERIES_FILE)

# --- Executa Busca ---
print("Executando busca...")
results_df = fusion.transform(queries_df)

# --- Salva CSV de submissão ---
print("Salvando resultados...")
submission_df = results_df[['qid', 'docno']].copy()
submission_df.columns = ['QueryId', 'EntityId']
submission_df = submission_df.sort_values(by='QueryId')
submission_df.to_csv(OUTPUT_FILE, index=False)
print(f"Arquivo salvo em: {OUTPUT_FILE} ({len(submission_df)} linhas)")

# --- Avalia se arquivos de treino estiverem disponíveis ---
if os.path.exists(TRAIN_QUERIES_FILE) and os.path.exists(QRELS_FILE):
    from pyterrier import Experiment

    train_queries = load_and_clean_queries(TRAIN_QUERIES_FILE)
    qrels = pd.read_csv(QRELS_FILE)
    qrels = qrels.rename(columns={
        'QueryId': 'qid',
        'EntityId': 'doc_id',
        'Relevance': 'relevance'
    })
    qrels['qid'] = qrels['qid'].astype(str)

    print("Executando avaliação...")
    eval = Experiment(
        [fusion],
        topics=train_queries,
        qrels=qrels,
        eval_metrics=[MAP, nDCG, Recall(10)]
    )
    print(eval)
