import os
import logging
import pandas as pd
import numpy as np
import pyterrier as pt
from tqdm import tqdm
import csv
import datetime
import time

from  index.dense_index import get_dense_retrieval_results
import query_processor.expanded_query_processor_test as expanded_query_processor_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
now_str = datetime.datetime.now().strftime("%m_%d_%H.%M")
FINAL_SUBMISSION_PATH = f'out/final_submission_{now_str}.csv'

def combine_results(sparse_results, dense_results, weight_sparse=0.7, weight_dense=0.3):
    """Combine sparse and dense retrieval results"""
    logging.info("Combining sparse and dense results...")
    
    # Normalize query IDs if needed
    sparse_results['qid'] = sparse_results['qid'].astype(str)
    dense_results['qid'] = dense_results['qid'].astype(str)
    
    # Normalize scores within each result set
    for df in [sparse_results, dense_results]:
        for qid in df['qid'].unique():
            mask = df['qid'] == qid
            max_score = df.loc[mask, 'score'].max()
            min_score = df.loc[mask, 'score'].min()
            if max_score > min_score:
                df.loc[mask, 'score'] = (df.loc[mask, 'score'] - min_score) / (max_score - min_score)
            else:
                df.loc[mask, 'score'] = 1.0
    
    # Merge results
    merged_results = pd.merge(
        sparse_results[['qid', 'docno', 'score']], 
        dense_results[['qid', 'docno', 'score']], 
        on=['qid', 'docno'], 
        how='outer',
        suffixes=('_sparse', '_dense')
    )
    
    # Fill NaN values with 0 (documents not found in one of the methods)
    merged_results = merged_results.fillna(0)
    
    # Combine scores with weights
    merged_results['final_score'] = (
        weight_sparse * merged_results['score_sparse'] + 
        weight_dense * merged_results['score_dense']
    )
    
    # Sort by query ID and final score
    merged_results = merged_results.sort_values(
        ['qid', 'final_score'], 
        ascending=[True, False]
    )
    
    # Create final results DataFrame
    final_results = merged_results[['qid', 'docno', 'final_score']].rename(
        columns={'final_score': 'score'}
    )
    
    # Take top 100 results per query
    final_results = final_results.groupby('qid').head(100)
    
    return final_results

def main():
    start_time = time.time()
    logging.info("Starting the enhanced IR pipeline...")
    
    # Initialize PyTerrier if not started
    if not pt.started():
        pt.init()
    
    # Step 2: Run dense retrieval
    logging.info("Step 2: Running dense retrieval...")
    #_, dense_test_results = run_dense_retrieval()
    _, dense_test_results = get_dense_retrieval_results()
    
    # Step 3: Run improved query processing and LTR
    logging.info("Step 3: Running improved query processing and LTR...")
    sparse_test_results = expanded_query_processor_test.run_enhanced_ltr_pipeline()
    
    # Step 4: Combine results and generate final submission
    logging.info("Step 4: Combining results and generating final submission...")
    final_results = combine_results(sparse_test_results, dense_test_results, weight_sparse=0.3, weight_dense=0.7)
    
    # Generate submission file
    os.makedirs(os.path.dirname(FINAL_SUBMISSION_PATH), exist_ok=True)
    with open(FINAL_SUBMISSION_PATH, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(final_results.iterrows(), total=len(final_results), desc="Generating final submission"):
            writer.writerow([str(row["qid"]).zfill(3), str(row["docno"]).zfill(7)])
    
    elapsed_time = time.time() - start_time
    logging.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    logging.info(f"Final submission saved to {FINAL_SUBMISSION_PATH}")

if __name__ == '__main__':
    main()