import pyterrier as pt
import pandas as pd
import json
import os
from tqdm import tqdm

# --- Configuration ---
CORPUS_FILE = 'corpus.jsonl'
QUERIES_FILE = 'test_queries.csv'
OUTPUT_FILE = 'submission.csv'
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
                    # Combine relevant fields into a single 'text' field for indexing
                    # Adjust field names if they differ in the actual corpus.jsonl
                    title = data.get('title', '')
                    keywords = data.get('keywords', '') # Assuming keywords is a string or list of strings
                    desc = data.get('text', '') # Assuming 'text' holds the description

                    # Handle potential list format for keywords
                    if isinstance(keywords, list):
                        keywords_str = ' '.join(keywords)
                    else:
                        keywords_str = str(keywords) if keywords else ''

                    # Combine fields, ensuring separation
                    combined_text = f"{title} {keywords_str} {desc}".strip()

                    # Use 'id' as docno, assuming it exists
                    doc_id = data.get('id')
                    if doc_id is None:
                        # print(f"Warning: Skipping line due to missing 'id': {line.strip()}")
                        continue # Skip if no ID

                    if not combined_text:
                         # print(f"Warning: Skipping doc {doc_id} due to empty combined text.")
                         continue # Skip if no text

                    docs.append({'docno': str(doc_id), 'text': combined_text})
                except json.JSONDecodeError:
                    # print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    pass # Silently skip invalid JSON
                except Exception as e:
                    # print(f"Warning: Skipping line due to error: {e} - Line: {line.strip()}")
                    pass # Silently skip other errors per line

        if not docs:
             raise ValueError("No documents could be processed from the corpus file. Please check the file format, content, and field names ('id', 'title', 'keywords', 'text').")

        return pd.DataFrame(docs)

    except FileNotFoundError:
        print(f"Error: Corpus file not found at {corpus_path}")
        print("Please download 'corpus.jsonl' from Kaggle and place it in the same directory as this script.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the corpus: {e}")
        exit(1)

# --- Main Script ---
if __name__ == "__main__":
    # 1. Initialize PyTerrier
    if not pt.started():
        print("Starting PyTerrier... (This may take a while during the first run)")
        try:
            # Increased memory allocation, adjust if needed
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], mem="4g")
        except Exception as e:
            print(f"Error initializing PyTerrier: {e}")
            print("Ensure Java (JDK 11 or later) is installed and configured correctly.")
            print("You might need to adjust the memory allocation ('mem' parameter) based on your system.")
            exit(1)

    # 2. Prepare and Index Corpus
    if not os.path.exists(INDEX_DIR + "/data.properties"):
        print("Index not found. Creating index...")
        corpus_df = prepare_corpus(CORPUS_FILE)
        print(f"Prepared {len(corpus_df)} documents for indexing.")
        if corpus_df.empty:
             print("Error: Corpus DataFrame is empty after preparation. Cannot create index.")
             exit(1)

        # Create an iterative indexer with specified meta fields
        indexer = pt.IterDictIndexer(INDEX_DIR, meta={'docno': 20}, meta_tags={'docno': 'docno'}, overwrite=True)

        print("Starting indexing process...")
        try:
            index_ref = indexer.index(corpus_df.to_dict(orient='records'))
            print(f"Indexing complete. Index created at: {index_ref.toString()}")
        except Exception as e:
             print(f"Error during indexing: {e}")
             print("Check available disk space and memory allocation.")
             exit(1)
    else:
        print(f"Using existing index at {INDEX_DIR}")
        index_ref = pt.IndexRef.of(INDEX_DIR + "/data.properties")

    # 3. Load Index and Define Retrieval Model
    try:
        index = pt.IndexFactory.of(index_ref)
        print(f"Index loaded successfully. Number of documents: {index.getCollectionStatistics().getNumberOfDocuments()}")
        # Define BM25 retrieval model as a starting point
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=MAX_RESULTS_PER_QUERY)
    except Exception as e:
        print(f"Error loading index or defining retrieval model: {e}")
        exit(1)

    # 4. Load Test Queries
    print(f"Loading queries from {QUERIES_FILE}...")
    try:
        queries_df = pd.read_csv(QUERIES_FILE, dtype={'QueryId': str})
        queries_df = queries_df.rename(columns={'QueryId': 'qid', 'Query': 'query'})
        print(f"Loaded {len(queries_df)} queries.")
    except FileNotFoundError:
        print(f"Error: Queries file not found at {QUERIES_FILE}")
        print("Please download 'test_queries.csv' from Kaggle and place it in the same directory as this script.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the queries file: {e}")
        exit(1)

    # 5. Perform Search
    print("Running search for all queries...")
    try:
        results_df = bm25.transform(queries_df)
        print(f"Search complete. Retrieved {len(results_df)} results overall.")
    except Exception as e:
        print(f"Error during search: {e}")
        exit(1)

    # 6. Format for Submission
    print(f"Formatting results for submission file: {OUTPUT_FILE}...")
    if 'docno' not in results_df.columns:
        print("Error: 'docno' column not found in search results. Check index configuration (meta fields).")
        exit(1)

    submission_df = results_df[['qid', 'docno']].copy()
    submission_df.rename(columns={'qid': 'QueryId', 'docno': 'EntityId'}, inplace=True)
    submission_df['EntityId'] = submission_df['EntityId'].astype(str)

    # Ensure correct sorting and limit (redundant if num_results worked, but safe) 
    submission_df = submission_df.sort_values(by=['QueryId'], ascending=True)
    # Grouping by QueryId and taking head might be needed if num_results wasn't perfectly enforced
    # submission_df = submission_df.groupby('QueryId').head(MAX_RESULTS_PER_QUERY) 

    # Save to CSV
    try:
        submission_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Submission file '{OUTPUT_FILE}' created successfully with {len(submission_df)} lines.")
        print("\n--- Instructions ---")
        print(f"1. Ensure '{CORPUS_FILE}' and '{QUERIES_FILE}' are in the same directory as this script.")
        print(f"2. Run the script using: python entity_search.py")
        print(f"3. The script will create an index in the '{INDEX_DIR}' directory (if it doesn't exist). This might take time and disk space.")
        print(f"4. The final submission file will be saved as '{OUTPUT_FILE}'.")
        print(f"5. Upload '{OUTPUT_FILE}' to the Kaggle competition: https://www.kaggle.com/t/5cff49e66f9d4e59a2c1f8e90db6d2ac")
        print("\n--- Next Steps & Tips for the Competition ---")
        print("- This script uses a basic BM25 model. Experiment with others (e.g., 'PL2', 'TF_IDF') in `pt.BatchRetrieve`.")
        print("- Customize indexing: Modify `pt.IterDictIndexer` parameters (e.g., `stemmer`, `stopwords`) or use `pt.TRECCollectionIndexer` for more control.")
        print("- Query Expansion: Explore techniques like Pseudo-Relevance Feedback (PRF) using `pt.rewrite.Bo1QueryExpansion(index)`.")
        print("- Fielded Search: If fields like 'title' are important, index them separately and use fielded query language or models.")
        print("- Learning to Rank: Use `train_queries.csv` and `train_qrels.csv` with models like LambdaMART (requires more setup).")
        print("- Check Logs: Pay attention to warnings during corpus processing and indexing.")
        print("- Memory: The `mem='4g'` in `pt.init` might need adjustment based on your system and corpus size.")

    except Exception as e:
        print(f"Error writing submission file: {e}")
        exit(1)

