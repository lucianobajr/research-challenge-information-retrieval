import pyterrier as pt
import pandas as pd
import json
import os
import re
from tqdm import tqdm
from pyterrier.terrier import Retriever

# --- Configuration ---
CORPUS_FILE = './data/corpus.jsonl'
QUERIES_FILE = './data/test_queries.csv'
OUTPUT_FILE = './out/submission.csv'
INDEX_DIR = './terrier_index'
MAX_RESULTS_PER_QUERY = 100

# --- Helper Function to Prepare Corpus ---
def prepare_corpus(corpus_path):
    """Reads the JSONL corpus and prepares it for PyTerrier indexing."""
    docs = []
    print(f"Reading corpus from {corpus_path}...")
    try:
        with open(corpus_path, 'r') as f:
            for line in tqdm(f, desc="Processing corpus"):
                try:
                    data = json.loads(line)
                    title = data.get('title', '')
                    keywords = data.get('keywords', '')
                    desc = data.get('text', '')

                    if isinstance(keywords, list):
                        keywords_str = ' '.join(keywords)
                    else:
                        keywords_str = str(keywords) if keywords else ''

                    combined_text = f"{title} {keywords_str} {desc}".strip()
                    doc_id = data.get('id')

                    if doc_id is None or not combined_text:
                        continue

                    docs.append({'docno': str(doc_id), 'text': combined_text})
                except:
                    pass

        if not docs:
            raise ValueError("No documents could be processed from the corpus file.")

        return pd.DataFrame(docs)

    except FileNotFoundError:
        print(f"Error: Corpus file not found at {corpus_path}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error reading the corpus: {e}")
        exit(1)

# --- Main Script ---
if __name__ == "__main__":
    # 1. Initialize PyTerrier
    print("Starting PyTerrier... (This may take a while during the first run)")
    try:
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    except Exception as e:
        print(f"Error initializing PyTerrier: {e}")
        exit(1)

    # 2. Prepare and Index Corpus
    if not os.path.exists(INDEX_DIR + "/data.properties"):
        print("Index not found. Creating index...")
        corpus_df = prepare_corpus(CORPUS_FILE)
        print(f"Prepared {len(corpus_df)} documents for indexing.")

        if corpus_df.empty:
            print("Error: Corpus is empty after preparation.")
            exit(1)

        indexer = pt.IterDictIndexer(INDEX_DIR, meta={'docno': 20}, overwrite=True)
        print("Starting indexing process...")
        try:
            index_ref = indexer.index(corpus_df.to_dict(orient='records'))
            print(f"Indexing complete at: {index_ref.toString()}")
        except Exception as e:
            print(f"Error during indexing: {e}")
            exit(1)
    else:
        print(f"Using existing index at {INDEX_DIR}")
        index_ref = pt.IndexRef.of(INDEX_DIR + "/data.properties")

    # 3. Load Index and Define Retrieval Model
    try:
        index = pt.IndexFactory.of(index_ref)
        print(f"Index loaded. Number of documents: {index.getCollectionStatistics().getNumberOfDocuments()}")
        bm25 = Retriever(index, wmodel="BM25", num_results=MAX_RESULTS_PER_QUERY)
    except Exception as e:
        print(f"Error loading index or model: {e}")
        exit(1)

   # 4. Load Test Queries
    print(f"Loading queries from {QUERIES_FILE}...")
    try:
        queries_df = pd.read_csv(QUERIES_FILE, dtype={'QueryId': str})
        queries_df = queries_df.rename(columns={'QueryId': 'qid', 'Query': 'query'})

        # Sanitize queries: remove all non-alphanumeric symbols except whitespace
        def clean_query(query):
            return re.sub(r"[^a-zA-Z0-9\s]", " ", query)

        queries_df['query'] = queries_df['query'].apply(clean_query)

        print(f"Loaded {len(queries_df)} queries.")
    except FileNotFoundError:
        print(f"Error: Queries file not found at {QUERIES_FILE}")
        exit(1)
    except Exception as e:
        print(f"Error reading queries file: {e}")
        exit(1)


    # 5. Perform Search
    print("Running search for all queries...")
    try:
        results_df = bm25.transform(queries_df)
        print(f"Search complete. Retrieved {len(results_df)} results.")
    except Exception as e:
        print(f"Error during search: {e}")
        exit(1)

    # 6. Format for Submission
    print(f"Formatting results for: {OUTPUT_FILE}...")
    if 'docno' not in results_df.columns:
        print("Error: 'docno' not in search results.")
        exit(1)

    submission_df = results_df[['qid', 'docno']].copy()
    submission_df.rename(columns={'qid': 'QueryId', 'docno': 'EntityId'}, inplace=True)
    submission_df['EntityId'] = submission_df['EntityId'].astype(str)
    submission_df = submission_df.sort_values(by=['QueryId'])

    try:
        submission_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Submission saved to: {OUTPUT_FILE} ({len(submission_df)} lines)")
        print("\n--- Tips ---")
        print("- You’re using BM25. Try other models like PL2 or DFR.")
        print("- Add PRF with `pt.rewrite.Bo1QueryExpansion(index)` if needed.")
        print("- Explore Learning to Rank with training data.")
    except Exception as e:
        print(f"Error writing submission file: {e}")
        exit(1)
