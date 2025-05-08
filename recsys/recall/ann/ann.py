from annoy import AnnoyIndex
import pandas as pd
import numpy as np
import os # Import os for path checking

# Define constants for dimensions (replace with actual values if different)
# These might be better passed during initialization or loaded from config
DEFAULT_QUERY_DIM = 32
DEFAULT_PRODUCT_DIM = 160
DEFAULT_TREE_DIR = "/Users/haozhezhang/Documents/Python Project/two_tower_search/recsys/recall/ann/trees" # Define base dir

class AnnoyRecall:
    """
    Manages Annoy indices for query and product embeddings
    to find approximate nearest neighbors.
    """
    def __init__(self,
                 query_tower_input_dim: int,
                 product_tower_input_dim: int,
                 # Update default paths using os.path.join
                 query_tree_path: str = os.path.join(DEFAULT_TREE_DIR, 'query.ann'),
                 product_tree_path: str = os.path.join(DEFAULT_TREE_DIR, 'product.ann')):
        """
        Initializes the AnnoyRecall manager.

        Args:
            query_tower_input_dim: Dimensionality of query vectors.
            product_tower_input_dim: Dimensionality of product vectors.
            query_tree_path: Path to save/load the query Annoy index.
            product_tree_path: Path to save/load the product Annoy index.
        """
        self.query_dim = query_tower_input_dim
        self.product_dim = product_tower_input_dim
        # Ensure the directory exists when initializing or before saving
        tree_dir = os.path.dirname(query_tree_path) # Get dir from one of the paths
        if not os.path.exists(tree_dir):
            print(f"Creating directory for Annoy trees: {tree_dir}")
            os.makedirs(tree_dir, exist_ok=True) # exist_ok=True prevents error if dir already exists

        self.q_tree_path = query_tree_path
        self.p_tree_path = product_tree_path

        # Annoy indices will be loaded on demand
        self.q_index: AnnoyIndex | None = None
        self.p_index: AnnoyIndex | None = None

    def build_and_save_indices(self,
                               df_query_embedding: pd.DataFrame,
                               df_product_embedding: pd.DataFrame,
                               num_trees: int = 100):
        """
        Builds Annoy indices from dataframes and saves them to disk.

        Args:
            df_query_embedding: DataFrame with query embeddings ('qid', 'q0', 'q1', ...).
            df_product_embedding: DataFrame with product embeddings ('pid', 'p0', 'p1', ...).
            num_trees: Number of trees to build for Annoy indices. Higher is more accurate but slower.
        """
        # Ensure the directory exists before saving (added for robustness)
        tree_dir = os.path.dirname(self.q_tree_path)
        if not os.path.exists(tree_dir):
            print(f"Creating directory for Annoy trees: {tree_dir}")
            os.makedirs(tree_dir, exist_ok=True)

        print("Building Query Index...")
        q = AnnoyIndex(self.query_dim, 'euclidean')
        query_cols = [f'q{i}' for i in range(self.query_dim)]
        for _, row in df_query_embedding.iterrows():
            key = int(row['qid'])
            vec = list(row[query_cols])
            q.add_item(key, vec)
        q.build(num_trees)
        q.save(self.q_tree_path)
        print(f"Query Index saved to {self.q_tree_path}")

        print("Building Product Index...")
        p = AnnoyIndex(self.product_dim, 'euclidean')
        product_cols = [f'p{i}' for i in range(self.product_dim)]
        for _, row in df_product_embedding.iterrows():
            key = int(row['pid'])
            vec = list(row[product_cols])
            p.add_item(key, vec)
        p.build(num_trees)
        p.save(self.p_tree_path)
        print(f"Product Index saved to {self.p_tree_path}")

        # Optionally load them into memory immediately after building
        # self.load_indices()

    def load_indices(self, force_reload: bool = False) -> bool:
        """
        Loads the Annoy indices from the specified paths into memory.

        Args:
            force_reload: If True, reload indices even if they are already in memory.

        Returns:
            True if both indices were loaded successfully, False otherwise.
        """
        loaded_q = False
        if self.q_index is None or force_reload:
            if os.path.exists(self.q_tree_path):
                print(f"Loading Query Index from {self.q_tree_path}...")
                self.q_index = AnnoyIndex(self.query_dim, 'euclidean')
                self.q_index.load(self.q_tree_path)
                loaded_q = True
                print("Query Index loaded.")
            else:
                print(f"Warning: Query index file not found at {self.q_tree_path}")
                self.q_index = None # Ensure it's None if load fails

        loaded_p = False
        if self.p_index is None or force_reload:
            if os.path.exists(self.p_tree_path):
                print(f"Loading Product Index from {self.p_tree_path}...")
                self.p_index = AnnoyIndex(self.product_dim, 'euclidean')
                self.p_index.load(self.p_tree_path)
                loaded_p = True
                print("Product Index loaded.")
            else:
                print(f"Warning: Product index file not found at {self.p_tree_path}")
                self.p_index = None # Ensure it's None if load fails

        return loaded_q and loaded_p

    def _ensure_product_index_loaded(self):
        """Ensures the product index is loaded, loading if necessary."""
        if self.p_index is None:
            # Try loading both, as load_indices handles both paths
            if not self.load_indices():
                 # Check specifically if p_index is still None after attempting load
                 if self.p_index is None:
                      raise FileNotFoundError(f"Product index file not found or failed to load from {self.p_tree_path}. Build it first.")
            if self.p_index is None: # Check again after trying to load
                 raise RuntimeError("Failed to load product index.")

    def _ensure_query_index_loaded(self):
        """Ensures the query index is loaded, loading if necessary."""
        if self.q_index is None:
            # Try loading both, as load_indices handles both paths
            if not self.load_indices():
                 # Check specifically if q_index is still None after attempting load
                 if self.q_index is None:
                      raise FileNotFoundError(f"Query index file not found or failed to load from {self.q_tree_path}. Build it first.")
            if self.q_index is None: # Check again after trying to load
                 raise RuntimeError("Failed to load query index.")


    def get_product_neighbors_by_vector(self, query_vector: list | np.ndarray, k: int, include_distances: bool = False) -> list[int] | list[tuple[int, float]]:
        """
        Finds the top-k nearest product neighbors for a given query vector.
        NOTE: This assumes the query_vector can be meaningfully compared
              in the product embedding space (e.g., same dimension or transformed).
              Annoy itself requires the vector dimension to match the index dimension (self.product_dim).
              If dimensions differ, this method will raise an error from Annoy.

        Args:
            query_vector: The query embedding vector (must match self.product_dim).
            k: The number of neighbors to retrieve.
            include_distances: Whether to return distances along with IDs.

        Returns:
            A list of product IDs, or a list of (product_ID, distance) tuples.
        """
        self._ensure_product_index_loaded()
        # Annoy will raise an error here if query_vector dim != self.product_dim
        results = self.p_index.get_nns_by_vector(query_vector, k, include_distances=include_distances)
        return results

    def get_query_neighbors_by_vector(self, query_vector: list | np.ndarray, k: int, include_distances: bool = False) -> list[int] | list[tuple[int, float]]:
        """
        Finds the top-k nearest query neighbors for a given query vector.

        Args:
            query_vector: The query embedding vector (must match self.query_dim).
            k: The number of neighbors to retrieve.
            include_distances: Whether to return distances along with IDs.

        Returns:
            A list of query IDs, or a list of (query_ID, distance) tuples.
        """
        self._ensure_query_index_loaded()
        # Annoy requires vector dim == self.query_dim
        results = self.q_index.get_nns_by_vector(query_vector, k, include_distances=include_distances)
        return results

    # --- Add similar methods for other operations if needed ---
    # e.g., get_product_neighbors_by_item, etc.

# --- Example Usage ---
if __name__ == '__main__':
    print("Loading data...")
    # Assume these files exist and have the correct format
    # Make sure paths are correct relative to where you run the script
    try:
        # Load product embeddings and add 'pid'
        df_product_meta = pd.read_parquet('versions/1/shopping_queries_dataset_products.parquet')[['product_id','product_title']].drop_duplicates()
        df_product_emb_vectors = pd.read_csv('versions/1/product_150k.csv') # Assuming this has product_id and p0..pN
        df_product_embedding = pd.merge(df_product_emb_vectors, df_product_meta, on='product_id').reset_index(drop=True)
        df_product_embedding['pid'] = df_product_embedding.index # Use index as pid

        # Load query embeddings and add 'qid'
        df_query_embedding_raw = pd.read_csv('versions/1/query_150k.csv') # Assuming this has query and q0..qN
        df_query_embedding_raw['qid'] = df_query_embedding_raw.index # Use index as qid

        # --- Determine dimensions AND Select Correct Columns ---
        # Use the intended dimension (DEFAULT_QUERY_DIM = 32)
        intended_query_dim = DEFAULT_QUERY_DIM
        intended_product_dim = DEFAULT_PRODUCT_DIM # Assuming this is correct (160)

        query_cols_to_use = ['qid', 'query'] + [f'q{i}' for i in range(intended_query_dim)]
        product_cols_to_use = ['pid', 'product_title', 'product_id'] + [f'p{i}' for i in range(intended_product_dim)] # Include needed meta cols

        # Filter DataFrames to only include necessary columns
        # Check if 'query' column exists before selecting
        if 'query' not in df_query_embedding_raw.columns:
             print("Error: 'query' column not found in query_150k.csv")
             exit()
        df_query_embedding = df_query_embedding_raw[query_cols_to_use]

        # Check if product embedding columns exist before selecting
        actual_product_cols_in_file = [f'p{i}' for i in range(intended_product_dim)]
        if not all(col in df_product_embedding.columns for col in actual_product_cols_in_file):
             missing_p_cols = [col for col in actual_product_cols_in_file if col not in df_product_embedding.columns]
             print(f"Error: Product embedding columns missing in product_150k.csv: {missing_p_cols}")
             exit()
        # Select only the necessary product columns (pid, p0-p159, title)
        df_product_embedding = df_product_embedding[['pid', 'product_title'] + actual_product_cols_in_file]


        print(f"Using query dimensions: {intended_query_dim}")
        print(f"Using product dimensions: {intended_product_dim}")

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure 'versions/1/product_150k.csv', 'versions/1/query_150k.csv', and 'versions/1/shopping_queries_dataset_products.parquet' exist.")
        exit() # Exit if data can't be loaded
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV/Parquet files: {e}")
        exit()


    # Create mapping dictionaries (outside the class)
    # Ensure 'pid' and 'qid' are suitable dictionary keys (e.g., integers)
    mp_product_dict = pd.Series(df_product_embedding.product_title.values, index=df_product_embedding.pid.astype(int)).to_dict()
    # Use bracket notation to access the 'query' column
    mp_query_dict = pd.Series(df_query_embedding['query'].values, index=df_query_embedding.qid.astype(int)).to_dict()

    print("Initializing AnnoyRecall...")
    # Initialize with the INTENDED dimensions
    ann_recall = AnnoyRecall(
        query_tower_input_dim=intended_query_dim,
        product_tower_input_dim=intended_product_dim
    )

    # --- Build indices (only if they don't exist or need rebuilding) ---
    # Pass the filtered DataFrames
    print("Building and saving indices...")
    ann_recall.build_and_save_indices(df_query_embedding, df_product_embedding)


    # --- Load indices (necessary before searching) ---
    if not ann_recall.load_indices():
        print("Failed to load indices. Exiting.")
        exit()

    # --- Perform a search ---
    if not df_query_embedding.empty:
        # Get an example query vector (e.g., the first one)
        example_query_row = df_query_embedding.iloc[0]
        # Select only the embedding columns for the vector
        query_embedding_cols = [f'q{i}' for i in range(intended_query_dim)]
        example_query_vector = list(example_query_row[query_embedding_cols])
        # Use bracket notation here as well for consistency and safety
        example_query_text = example_query_row['query']
        example_query_id = int(example_query_row['qid'])

        # --- Search for similar QUERIES using the QUERY index ---
        print(f"\nFinding neighbors for Query ID {example_query_id}: '{example_query_text}'")
        top_k = 10
        try:
            # Call the new method to search the query index
            neighbor_qids = ann_recall.get_query_neighbors_by_vector(example_query_vector, k=top_k)

            print(f"Top {top_k} Similar Query Neighbors (IDs): {neighbor_qids}")
            print("Corresponding Query Texts:")
            for qid in neighbor_qids:
                # Use the query mapping dictionary
                print(f"  - ID {qid}: {mp_query_dict.get(qid, 'Query text not found')}")

        except (FileNotFoundError, RuntimeError, Exception) as e:
             print(f"Error during search: {e}")
    else:
        print("Query dataframe is empty, cannot perform search.")
