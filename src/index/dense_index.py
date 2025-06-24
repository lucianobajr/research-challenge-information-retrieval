import os
import logging
import pandas as pd
import numpy as np
import pyterrier as pt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import json

from constants.paths import CORPUS_PATH, DENSE_INDEX_DIR, DENSE_RESULTS_PATH, EMBEDDINGS_DIR, TEST_QUERIES_PATH, TRAIN_QUERIES_PATH

# Configuração do log
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Criação de diretórios necessários
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(DENSE_INDEX_DIR, exist_ok=True)
os.makedirs(DENSE_RESULTS_PATH, exist_ok=True)

# Inicializa o PyTerrier
if not pt.started():
    pt.init()

class DenseRetrieval:
    def __init__(self, model_name='sentence-transformers/msmarco-MiniLM-L6-cos-v5'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.doc_ids = None

    def encode_documents(self, documents, batch_size=32):
        """Codifica os documentos usando o modelo denso"""
        logging.info(f"Codificando {len(documents)} documentos...")

        embeddings = []
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_queries(self, queries):
        """Codifica as consultas usando o modelo denso"""
        logging.info(f"Codificando {len(queries)} consultas...")
        return self.model.encode(
            queries,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def build_index(self, embeddings):
        """Cria o índice FAISS para busca vetorial"""
        logging.info("Construindo índice FAISS...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Produto interno para similaridade de cosseno
        self.index.add(embeddings)
        return self.index

    def save_index(self, index_path):
        """Salva o índice FAISS no disco"""
        logging.info(f"Salvando índice em {index_path}...")
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path):
        """Carrega o índice FAISS do disco"""
        logging.info(f"Carregando índice de {index_path}...")
        self.index = faiss.read_index(index_path)

    def search(self, query_embeddings, k=100):
        """Realiza busca no índice FAISS"""
        logging.info(
            f"Buscando top {k} documentos para {len(query_embeddings)} consultas...")
        scores, doc_indices = self.index.search(query_embeddings, k)

        results = []
        for i, (query_scores, query_doc_indices) in enumerate(zip(scores, doc_indices)):
            for j, (score, doc_idx) in enumerate(zip(query_scores, query_doc_indices)):
                if doc_idx < 0:  # FAISS pode retornar -1 quando não encontra documentos suficientes
                    continue
                results.append({
                    'query_idx': i,
                    'doc_id': self.doc_ids[doc_idx],
                    'rank': j + 1,
                    'score': float(score)
                })

        return results


def load_documents():
    """Carrega e pré-processa documentos do corpus"""
    logging.info("Carregando documentos do corpus...")

    docs = []
    doc_ids = []

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                data = doc if 'id' in doc else doc.get('root', {})
                doc_id = data.get('id')

                if not doc_id:
                    continue

                title = data.get('title', '')
                text = data.get('text', '')

                # Junta título e corpo para gerar o embedding
                content = f"{title} {text}"

                docs.append(content)
                doc_ids.append(doc_id)
            except json.JSONDecodeError:
                continue

    return docs, doc_ids


def process_queries(query_path):
    """Carrega e processa consultas a partir de CSV"""
    logging.info(f"Carregando consultas de {query_path}...")
    queries_df = pd.read_csv(query_path)
    query_texts = queries_df['Query'].tolist()
    query_ids = queries_df['QueryId'].astype(str).tolist()

    return query_texts, query_ids


def create_pyterrier_results(search_results, query_ids):
    """Converte resultados do FAISS para o formato esperado pelo PyTerrier"""
    results_list = []

    for result in search_results:
        query_idx = result['query_idx']
        results_list.append({
            'qid': query_ids[query_idx],
            'docno': result['doc_id'],
            'rank': result['rank'],
            'score': result['score']
        })

    return pd.DataFrame(results_list)


def create_dense_index(dense_retriever, embeddings_path, doc_ids_path, index_path):
    """Cria e salva o índice denso a partir do corpus"""
    docs, doc_ids = load_documents()
    dense_retriever.doc_ids = doc_ids

    # Codifica os documentos
    embeddings = dense_retriever.encode_documents(docs)

    # Salva os embeddings e os IDs
    np.save(embeddings_path, embeddings)
    np.save(doc_ids_path, doc_ids)

    # Cria e salva o índice
    dense_retriever.build_index(embeddings)
    dense_retriever.save_index(index_path)


def run_dense_retrieval():
    """Executa a recuperação densa (treinamento e teste)"""
    dense_retriever = DenseRetrieval()
    index_path = os.path.join(DENSE_INDEX_DIR, 'faiss_index')
    embeddings_path = os.path.join(EMBEDDINGS_DIR, 'doc_embeddings.npy')
    doc_ids_path = os.path.join(EMBEDDINGS_DIR, 'doc_ids.npy')

    # Verifica se precisa construir o índice
    if not os.path.exists(index_path) or not os.path.exists(embeddings_path):
        create_dense_index(dense_retriever, embeddings_path, doc_ids_path, index_path)
    else:
        # Carrega índice e IDs previamente salvos
        dense_retriever.load_index(index_path)
        dense_retriever.doc_ids = np.load(doc_ids_path)

    # Codifica e busca para queries de treino
    train_query_texts, train_query_ids = process_queries(TRAIN_QUERIES_PATH)
    train_query_embeddings = dense_retriever.encode_queries(train_query_texts)
    train_results = dense_retriever.search(train_query_embeddings, k=100)
    train_df = create_pyterrier_results(train_results, train_query_ids)
    train_df.to_csv(os.path.join(DENSE_RESULTS_PATH, 'dense_train_results.csv'), index=False)

    # Codifica e busca para queries de teste
    test_query_texts, test_query_ids = process_queries(TEST_QUERIES_PATH)
    test_query_embeddings = dense_retriever.encode_queries(test_query_texts)
    test_results = dense_retriever.search(test_query_embeddings, k=100)
    test_df = create_pyterrier_results(test_results, test_query_ids)
    test_df.to_csv(os.path.join(DENSE_RESULTS_PATH, 'dense_test_results.csv'), index=False)

    logging.info("Resultados da recuperação densa salvos no diretório dense_results")
    return train_df, test_df


def get_dense_retrieval_results():
    """
    Recupera resultados usando um índice denso já existente.
    Retorna:
        tupla: (train_df, test_df) com os resultados de recuperação
    """
    logging.info("Carregando índice denso existente...")

    dense_retriever = DenseRetrieval()
    index_path = os.path.join(DENSE_INDEX_DIR, 'faiss_index')
    doc_ids_path = os.path.join(EMBEDDINGS_DIR, 'doc_ids.npy')

    # Carrega índice e IDs
    dense_retriever.load_index(index_path)
    dense_retriever.doc_ids = np.load(doc_ids_path)
    logging.info(f"Índice carregado com {dense_retriever.index.ntotal} vetores")

    # Busca para treino
    logging.info("Processando consultas de treino...")
    train_query_texts, train_query_ids = process_queries(TRAIN_QUERIES_PATH)
    train_query_embeddings = dense_retriever.encode_queries(train_query_texts)
    logging.info("Buscando resultados para treino...")
    train_results = dense_retriever.search(train_query_embeddings, k=100)
    train_df = create_pyterrier_results(train_results, train_query_ids)
    train_df.to_csv(os.path.join(DENSE_RESULTS_PATH, 'dense_train_results.csv'), index=False)

    # Busca para teste
    logging.info("Processando consultas de teste...")
    test_query_texts, test_query_ids = process_queries(TEST_QUERIES_PATH)
    test_query_embeddings = dense_retriever.encode_queries(test_query_texts)
    logging.info("Buscando resultados para teste...")
    test_results = dense_retriever.search(test_query_embeddings, k=100)
    test_df = create_pyterrier_results(test_results, test_query_ids)
    test_df.to_csv(os.path.join(DENSE_RESULTS_PATH, 'dense_test_results.csv'), index=False)

    logging.info("Resultados da recuperação densa salvos com sucesso.")
    return train_df, test_df


if __name__ == '__main__':
    run_dense_retrieval()
