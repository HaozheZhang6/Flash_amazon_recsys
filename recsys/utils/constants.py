# Ensure these paths are correct relative to your execution environment
PRODUCT_EMBEDDING_CSV = 'data/product_150k.csv'
PRODUCT_META_PARQUET = 'data/shopping_queries_dataset_products.parquet'
MODEL_PATH_DICT = {
    "point_wise": "models/two_towers/point_wise/final_model.pt",
    "pair_wise": "models/two_towers/pair_wise/final_model.pt",
    "list_wise": "models/two_towers/list_wise/final_model.pt"
}
MODEL_METHOD = "list_wise"
INPUT_DIM_Q = 32
INPUT_DIM_P = 160
HIDDEN_Q = 64
HIDDEN_P = 64
EMBED = 32
ANN_QUERY_DIM = 32
ANN_PRODUCT_DIM = 160

FINAL_EMBEDDINGS_CSV = 'data/final_product_embeddings.csv'
RESULTS_DIR = 'results'

# Add these constants
VECTOR_SEARCH_N_REGIONS = 1000
VECTOR_SEARCH_N_NEAREST_ZONES = 5  # Number of nearest zones to search

NUM_PRODUCT_TREES = 10