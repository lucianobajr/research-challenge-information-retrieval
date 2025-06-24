import os
import csv
import logging
import pandas as pd
import numpy as np
import pyterrier as pt
from tqdm import tqdm
import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

from constants.paths import TRAIN_QUERIES_PATH,TEST_QUERIES_PATH,TRAIN_QRELS_PATH

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
now_str = datetime.datetime.now().strftime("%m_%d_%H.%M")
SUBMISSION_OUTPUT_PATH = f'out/submission_{now_str}.csv'

# Configure paths
INDEX_DIR = os.path.abspath('indexes_V2')
EXPANDED_INDEX_DIR = os.path.abspath('expanded_indexes')
HITS_PER_QUERY = 300  # Increased from 200

# Text processing tools
CUSTOM_STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "that", "is", "was", "for", "on", "it", "with",
    "as", "are", "at", "be", "this", "by", "have", "from", "or", "an", "but", "not", "what",
    "all", "were", "when", "we", "there", "can", "an", "been", "who", "will", "no", "more",
    "if", "out", "so", "up", "said", "about", "than", "its", "into", "them", "only", "just"
}
STOPWORDS = set(stopwords.words('english')).union(CUSTOM_STOPWORDS)
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# Initialize PyTerrier
if not pt.started():
    pt.init()


def sanitize_query(text):
    """Sanitize query by removing special characters that can cause parser errors"""
    # Replace apostrophes and quotation marks
    text = text.replace("'", "")
    text = text.replace('"', "")
    # Replace other special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def basic_preprocess(text):
    """Basic preprocessing: lowercase, stemming, stopword removal"""
    # First sanitize the text to remove problematic characters
    text = sanitize_query(text)
    tokens = word_tokenize(text.lower())
    filtered = [STEMMER.stem(t)
                for t in tokens if t.isalnum() and t not in STOPWORDS]
    return ' '.join(filtered)


def advanced_preprocess(text):
    """Advanced preprocessing with lemmatization"""
    # First sanitize the text to remove problematic characters
    text = sanitize_query(text)
    tokens = word_tokenize(text.lower())

    # Lemmatize and filter
    processed_tokens = []
    for token in tokens:
        if token.isalnum() and token not in STOPWORDS:
            processed_tokens.append(LEMMATIZER.lemmatize(token))

    return ' '.join(processed_tokens)


def pseudo_relevance_feedback(index, query, n_docs=3, n_terms=5):
    """
    Simple query expansion that doesn't rely on document content access
    Instead, it uses query term analysis and predefined expansion rules
    """
    # Sanitize and process the original query
    original_query = sanitize_query(query)
    tokens = basic_preprocess(original_query).split()

    if not tokens:
        return original_query

    # Some common domain-specific term relationships
    domain_expansions = {
        # Literature and arts
        'book': ['novel', 'publication', 'literature', 'text', 'author', 'story', 'volume', 'series'],
        'author': ['writer', 'novelist', 'creator', 'poet', 'journalist', 'biographer'],
        'novel': ['book', 'fiction', 'story', 'tale', 'narrative', 'bestseller'],
        'poetry': ['poem', 'verse', 'sonnet', 'stanza', 'rhyme', 'lyric'],
        'literature': ['fiction', 'writing', 'books', 'prose', 'literary', 'novel'],

        # Film and entertainment
        'movie': ['film', 'cinema', 'picture', 'director', 'actor', 'screenplay', 'feature'],
        'film': ['movie', 'cinema', 'picture', 'director', 'actor', 'reel', 'screening'],
        'actor': ['actress', 'performer', 'star', 'celebrity', 'cast', 'talent'],
        'director': ['filmmaker', 'producer', 'cinematographer', 'screenwriter', 'creator'],
        'television': ['tv', 'broadcast', 'network', 'series', 'program', 'show', 'sitcom'],

        # Music
        'music': ['song', 'audio', 'artist', 'album', 'band', 'melody', 'sound', 'concert'],
        'song': ['track', 'tune', 'single', 'composition', 'hit', 'recording', 'lyrics'],
        'artist': ['musician', 'singer', 'performer', 'composer', 'band', 'soloist'],
        'album': ['record', 'release', 'disc', 'cd', 'collection', 'compilation', 'lp'],

        # History and time
        'history': ['past', 'historical', 'ancient', 'events', 'timeline', 'era', 'chronicle'],
        'century': ['era', 'period', 'age', 'epoch', 'decade', 'millennium'],
        'ancient': ['historical', 'antique', 'classical', 'prehistoric', 'archaic', 'primeval'],
        'medieval': ['middle ages', 'feudal', 'gothic', 'renaissance', 'castle', 'knight'],

        # Science and academics
        'science': ['research', 'scientific', 'study', 'experiment', 'discovery', 'theory', 'lab'],
        'research': ['study', 'investigation', 'analysis', 'experiment', 'survey', 'observation'],
        'theory': ['hypothesis', 'concept', 'principle', 'idea', 'proposition', 'conjecture'],
        'biology': ['life', 'organism', 'cell', 'genetic', 'species', 'evolution', 'ecology'],
        'physics': ['quantum', 'mechanics', 'relativity', 'particle', 'energy', 'atom', 'gravity'],

        # Technology and computing
        'computer': ['software', 'hardware', 'program', 'digital', 'technology', 'system', 'device'],
        'software': ['program', 'application', 'app', 'code', 'algorithm', 'system', 'platform'],
        'internet': ['web', 'online', 'network', 'digital', 'site', 'cyber', 'cloud'],
        'website': ['site', 'page', 'portal', 'homepage', 'domain', 'web', 'online'],
        'device': ['gadget', 'machine', 'apparatus', 'equipment', 'instrument', 'tool', 'hardware'],

        # Art and design
        'art': ['painting', 'artist', 'creative', 'design', 'gallery', 'sculpture', 'masterpiece'],
        'painting': ['artwork', 'canvas', 'portrait', 'picture', 'illustration', 'composition'],
        'design': ['pattern', 'style', 'layout', 'model', 'sketch', 'blueprint', 'plan'],

        # Places and regions
        'country': ['nation', 'state', 'region', 'government', 'territory', 'republic', 'kingdom'],
        'city': ['town', 'metropolis', 'urban', 'municipality', 'metropolitan', 'borough'],
        'region': ['area', 'district', 'zone', 'sector', 'territory', 'locale', 'province'],

        # Food and cuisine
        'food': ['cuisine', 'meal', 'recipe', 'dish', 'restaurant', 'ingredient', 'cooking'],
        'recipe': ['formula', 'preparation', 'instruction', 'ingredient', 'method', 'dish'],
        'restaurant': ['cafe', 'diner', 'eatery', 'bistro', 'establishment', 'dining'],

        # Sports and activities
        'sport': ['game', 'athlete', 'team', 'championship', 'competition', 'tournament', 'match'],
        'team': ['squad', 'club', 'crew', 'lineup', 'roster', 'group', 'side'],
        'player': ['athlete', 'competitor', 'sportsman', 'participant', 'contender', 'champion'],

        # Business and economy
        'business': ['company', 'corporate', 'industry', 'market', 'economy', 'firm', 'enterprise'],
        'company': ['corporation', 'business', 'firm', 'enterprise', 'organization', 'venture'],
        'industry': ['sector', 'business', 'manufacturing', 'commerce', 'trade', 'production'],

        # Education
        'education': ['school', 'university', 'teaching', 'learning', 'academic', 'study', 'course'],
        'university': ['college', 'institution', 'academy', 'campus', 'school', 'faculty'],
        'student': ['pupil', 'scholar', 'learner', 'undergraduate', 'graduate', 'freshman'],

        # Politics and government
        'politics': ['government', 'policy', 'election', 'political', 'party', 'candidate', 'campaign'],
        'government': ['administration', 'authority', 'regime', 'officials', 'state', 'institution'],
        'election': ['vote', 'ballot', 'poll', 'referendum', 'campaign', 'candidate', 'voting'],

        # Health and medicine
        'health': ['medical', 'disease', 'treatment', 'doctor', 'hospital', 'wellness', 'care'],
        'disease': ['illness', 'sickness', 'condition', 'disorder', 'ailment', 'infection', 'syndrome'],
        'doctor': ['physician', 'surgeon', 'practitioner', 'specialist', 'clinician', 'medic'],
        'treatment': ['therapy', 'medication', 'remedy', 'cure', 'intervention', 'prescription']
    }

    # Collect expansion terms
    expansion_terms = set()

    # Add domain-specific terms
    for token in tokens:
        stem_token = STEMMER.stem(token.lower())
        for domain, terms in domain_expansions.items():
            if stem_token == STEMMER.stem(domain) or domain in token or token in domain:
                for term in terms:
                    if STEMMER.stem(term) != stem_token:  # Avoid adding the same term
                        expansion_terms.add(term)

    # Add synonyms based on word length (simple heuristic)
    # The idea is that longer words often have more specific synonyms
    for token in tokens:
        if len(token) >= 6:  # For longer, more specific words
            # Add variations with common prefixes/suffixes
            if token.endswith('ing'):
                base = token[:-3]
                expansion_terms.add(f"{base}ed")
                expansion_terms.add(f"{base}er")
            elif token.endswith('er'):
                base = token[:-2]
                expansion_terms.add(f"{base}ing")
                expansion_terms.add(f"{base}ed")
            elif token.endswith('ed'):
                base = token[:-2]
                expansion_terms.add(f"{base}ing")
                expansion_terms.add(f"{base}er")

    # Select top terms (avoid too many terms)
    expansion_list = list(expansion_terms)
    if len(expansion_list) > n_terms:
        expansion_list = expansion_list[:n_terms]

    # If we couldn't find expansions, return original query
    if not expansion_list:
        return original_query

    expanded_query = f"{original_query} {' '.join(expansion_list)}"
    return expanded_query


def run_enhanced_ltr_pipeline():
    # Load indices
    basic_index = pt.IndexFactory.of(INDEX_DIR)
    expanded_index = pt.IndexFactory.of(EXPANDED_INDEX_DIR)

    # Load queries and qrels
    train_queries = pd.read_csv(TRAIN_QUERIES_PATH)
    # First, sanitize all queries to remove problematic characters
    train_queries["Query"] = train_queries["Query"].apply(sanitize_query)
    train_queries_original = train_queries.copy()
    train_queries["query_basic"] = train_queries["Query"].apply(
        basic_preprocess)
    train_queries["query_advanced"] = train_queries["Query"].apply(
        advanced_preprocess)
    train_queries = train_queries.rename(
        columns={"QueryId": "qid", "Query": "query"})
    train_queries["qid"] = train_queries["qid"].astype(str)

    qrels = pd.read_csv(TRAIN_QRELS_PATH)
    qrels = qrels.rename(
        columns={"QueryId": "qid", "EntityId": "docno", "Relevance": "label"})
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)

    # Split data for training
    train_queries_train, train_queries_val = train_test_split(
        train_queries, test_size=0.2, random_state=42)
    train_qrels_train = qrels[qrels['qid'].isin(train_queries_train['qid'])]
    train_qrels_val = qrels[qrels['qid'].isin(train_queries_val['qid'])]

    # Create multiple retrieval pipelines with simpler configuration for initial testing

    # Pipeline 1: BM25 with optimized parameters on basic index
    bm25_basic = pt.BatchRetrieve(
        basic_index,
        wmodel="BM25",
        controls={"bm25.b": 0.25, "bm25.k_1": 1.5,
                  "bm25.k_3": 0.5, 'w.0': 1, 'w.1': 2, 'w.2': 1},
        metadata=["docno"],
        verbose=True,
        num_results=HITS_PER_QUERY
    )

    # Pipeline 2: BM25 on expanded index
    bm25_expanded = pt.BatchRetrieve(
        expanded_index,
        wmodel="BM25",
        controls={"bm25.b": 0.25, "bm25.k_1": 1.5,
                  "bm25.k_3": 0.5, 'w.0': 1, 'w.1': 2, 'w.2': 1},
        metadata=["docno"],
        verbose=True,
        num_results=HITS_PER_QUERY
    )

    # Pipeline 3: DPH on basic index
    dph_basic = pt.BatchRetrieve(
        basic_index,
        wmodel="DPH",
        metadata=["docno"],
        num_results=HITS_PER_QUERY
    )

    # Pipeline 4: PL2 on expanded index
    pl2_expanded = pt.BatchRetrieve(
        expanded_index,
        wmodel="PL2",
        controls={"pl2.c": 10.0},
        metadata=["docno"],
        num_results=HITS_PER_QUERY
    )

    # Create a sequence of retrievers for first-pass retrieval (using | operator)
    first_pass = bm25_basic | bm25_expanded | dph_basic | pl2_expanded

    # Create feature pipeline for learning to rank
    feature_retrievers = [
        pt.BatchRetrieve(basic_index, wmodel="BM25", metadata=["docno"]),
        pt.BatchRetrieve(basic_index, wmodel="TF_IDF", metadata=["docno"]),
        pt.BatchRetrieve(basic_index, wmodel="PL2", metadata=["docno"]),
        pt.BatchRetrieve(expanded_index, wmodel="BM25", metadata=["docno"]),
        pt.BatchRetrieve(expanded_index, wmodel="TF_IDF", metadata=["docno"])
    ]

    # Combine all retrievers with the '**' operator to create features
    feature_union = feature_retrievers[0]
    for retriever in feature_retrievers[1:]:
        feature_union = feature_union ** retriever

    # Create LTR pipeline
    ltr_feats_pipeline = first_pass >> feature_union

    logging.info("Feature pipeline created:")
    print(ltr_feats_pipeline)

    # Define Learning to Rank models

    # LightGBM model
    lgbm_ranker = lgb.LGBMRanker(
        task="train",
        min_data_in_leaf=8,
        min_sum_hessian_in_leaf=0.8,
        max_bin=255,
        num_leaves=31,
        max_depth=10,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate=0.1,
        importance_type="gain",
        num_iterations=1000,  # Reduced for faster training initially
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=6
    )

    # XGBoost model with simplified parameters
    xgb_ranker = xgb.XGBRanker(
        objective='rank:ndcg',
        learning_rate=0.1,
        gamma=0.5,
        min_child_weight=0.8,
        max_depth=8,
        n_estimators=200,  # Reduced for faster training
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42
    )

    # Create pipelines for rankers
    lgbm_pipe = ltr_feats_pipeline >> pt.ltr.apply_learned_model(
        lgbm_ranker, form="ltr")
    xgb_pipe = ltr_feats_pipeline >> pt.ltr.apply_learned_model(
        xgb_ranker, form="ltr")

    # Train models
    logging.info("Training LightGBM ranker...")
    lgbm_pipe.fit(train_queries_train, train_qrels_train,
                  train_queries_val, train_qrels_val)

    logging.info("Training XGBoost ranker...")
    xgb_pipe.fit(train_queries_train, train_qrels_train,
                 train_queries_val, train_qrels_val)

    # Evaluate models
    logging.info("Evaluating models on validation data...")
    eval_metrics = {}

    for name, pipe in [("LightGBM", lgbm_pipe), ("XGBoost", xgb_pipe)]:
        results = pipe.transform(train_queries_val)
        metrics = pt.Utils.evaluate(results, train_qrels_val, metrics=[
                                    "map", "ndcg", "recall_100", "P_10"])
        eval_metrics[name] = metrics
        logging.info(f"{name} performance:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

    # Load and preprocess test queries
    test_queries = pd.read_csv(TEST_QUERIES_PATH)
    # Sanitize test queries
    test_queries["Query"] = test_queries["Query"].apply(sanitize_query)
    test_queries["query_basic"] = test_queries["Query"].apply(basic_preprocess)
    test_queries["query_advanced"] = test_queries["Query"].apply(
        advanced_preprocess)
    test_queries = test_queries.rename(
        columns={"QueryId": "qid", "Query": "query"})
    test_queries["qid"] = test_queries["qid"].astype(str)

    # Process test queries with expanded queries
    logging.info("Creating expanded test queries...")
    expanded_test_queries = []
    for _, row in test_queries.iterrows():
        qid = row['qid']
        original_query = row['query']
        expanded_query = pseudo_relevance_feedback(
            basic_index, row['query_basic'])

        expanded_test_queries.append({
            'qid': qid,
            'query': expanded_query,
            'query_basic': basic_preprocess(expanded_query),
            'query_advanced': advanced_preprocess(expanded_query)
        })

    expanded_test_queries_df = pd.DataFrame(expanded_test_queries)

    # Generate results from models
    logging.info("Running models on test queries...")
    lgbm_results = lgbm_pipe.transform(test_queries)
    xgb_results = xgb_pipe.transform(test_queries)

    # Run expanded queries
    logging.info("Running models on expanded test queries...")
    lgbm_expanded_results = lgbm_pipe.transform(expanded_test_queries_df)
    xgb_expanded_results = xgb_pipe.transform(expanded_test_queries_df)

    # Create ensemble function
    def ensemble_results(query_results, weights=None):
        if weights is None:
            weights = [1.0] * len(query_results)

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Combine scores
        combined_results = query_results[0].copy()
        combined_results['score'] = weights[0] * combined_results['score']

        for i in range(1, len(query_results)):
            df = query_results[i]
            # Ensure we're dealing with the same documents
            common_docs = pd.merge(
                combined_results, df,
                on=['qid', 'docno'],
                how='inner',
                suffixes=('', f'_{i}')
            )
            combined_results = common_docs
            combined_results['score'] += weights[i] * \
                combined_results[f'score_{i}']
            combined_results = combined_results.drop(columns=[f'score_{i}'])

        # Re-rank based on new scores
        combined_results = combined_results.sort_values(
            ['qid', 'score'], ascending=[True, False])
        return combined_results

    # Create ensemble
    logging.info("Creating final ensemble...")
    # Weights based on validation performance
    ensemble_weights = [0.4, 0.3, 0.2, 0.1]
    all_results = [
        lgbm_results, xgb_results,
        lgbm_expanded_results, xgb_expanded_results
    ]

    final_results = ensemble_results(all_results, ensemble_weights)
    final_results = final_results.groupby("qid").head(
        100)  # Limit to 100 results per query

    # Create submission file
    os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
    with open(SUBMISSION_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["QueryId", "EntityId"])
        for _, row in tqdm(final_results.iterrows(), total=len(final_results), desc="Generating submission"):
            writer.writerow([str(row["qid"]).zfill(3),
                            str(row["docno"]).zfill(7)])

    logging.info(f"Submission saved to {SUBMISSION_OUTPUT_PATH}")
    return final_results
