import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass

@dataclass
class TrainingData:
    """Container for different types of training data"""
    point_wise: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray] = None
    pair_wise: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None
    list_wise: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = None

def calculate_and_save_sampling_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sampling weights for products based on both E and I labels
    """
    # Calculate frequency for each product (both E and I)
    product_freq = df[df['esci_label'].isin(['E', 'I'])].groupby('product_id').size()
    
    # Calculate weights (freq ^ 0.75)
    weights = product_freq ** 0.75
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Add weights to the dataset
    df['sampling_weight'] = df['product_id'].map(weights).fillna(0)
    
    return df

def prepare_and_save_dataset() -> pd.DataFrame:
    """
    Prepare the dataset with pre-calculated weights and save it
    """
    print("Preparing and saving dataset...")
    
    # Load product embedding and title
    df_product_embedding = pd.merge(
        pd.read_csv('versions/1/product_150k.csv'),
        pd.read_parquet('versions/1/shopping_queries_dataset_products.parquet')[
            ['product_id', 'product_title']
        ].drop_duplicates(),
        on=['product_id']
    ).reset_index(drop=True)
    
    df_product_embedding['pid'] = range(0, df_product_embedding.shape[0])
    
    # Load query embedding and title
    df_dataset = pd.merge(
        pd.read_csv('versions/1/dataset_150k.csv'),
        df_product_embedding[['product_id', 'product_title']].drop_duplicates(),
        on=['product_id']
    )
    
    # Calculate and add sampling weights
    df_dataset = calculate_and_save_sampling_weights(df_dataset)
    
    # Save the processed dataset
    df_dataset.to_csv('versions/1/dataset_150k_processed.csv', index=False)
    print("Dataset saved successfully!")
    
    return df_dataset


def sample_negative_products(
    positive_products: pd.DataFrame,
    all_products: pd.DataFrame,
    n_samples: int,
) -> pd.DataFrame:
    """
    Simple weighted random sampling of negative products
    """
    return all_products.sample(
        n=n_samples,
        replace=True,
        weights='sampling_weight'
    )

def prepare_point_wise_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
    """
    Prepare data for point-wise training
    For each positive sample (E), create a negative sample with the same query
    """
    # Get positive samples (clicked)
    positive_samples = df[df['esci_label'] == 'E'].copy()
    
    if len(positive_samples) == 0:
        raise ValueError("No positive samples found in the dataset")
    
    # For each positive sample, create a negative sample with the same query
    negative_samples = []
    for _, pos_row in positive_samples.iterrows():
        # Sample one negative product
        neg_product = sample_negative_products(
            positive_samples,
            df,
            n_samples=1
        ).iloc[0]
        
        # Create negative sample with same query
        neg_sample = neg_product.copy()
        neg_sample['query'] = pos_row['query']
        neg_sample['esci_label'] = 'I'  # Mark as not clicked
        negative_samples.append(neg_sample)
    
    # Combine positive and negative samples
    return pd.concat([positive_samples, pd.DataFrame(negative_samples)]).reset_index(drop=True), None

def prepare_pair_wise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for pair-wise training
    Randomly sample negative products and assign them to queries
    """
    # Get positive samples (clicked)
    positive_samples = df[df['esci_label'] == 'E'].copy()
    
    if len(positive_samples) == 0:
        raise ValueError("No positive samples found in the dataset")
    
    # Sample negative products (3x the number of positive samples)
    negative_samples = sample_negative_products(
        positive_samples,
        df,
        n_samples=len(positive_samples) * 3
    )
    
    # Create pairs by combining positive samples with negative samples
    pairs = pd.DataFrame({
        'query': positive_samples['query'].repeat(3),
        'product_id_positive': positive_samples['product_id'].repeat(3),
        'product_id_negative': negative_samples['product_id'].values,
        **{f'q{i}': positive_samples[f'q{i}'].repeat(3).values for i in range(32)},
        **{f'p{i}_positive': positive_samples[f'p{i}'].repeat(3).values for i in range(160)},
        **{f'p{i}_negative': negative_samples[f'p{i}'].values for i in range(160)}
    })
    
    return pairs

def prepare_list_wise_data(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Prepare data for list-wise training
    Each batch contains query-product-reward triplets
    """
    # Get all samples with rewards
    list_wise_data = df.copy()
    list_wise_data['reward'] = list_wise_data['esci_label'].map({
        'E': 1.0,  # Clicked
        'I': 0.5,  # Shown but not clicked
        'S': 0.0   # Not shown
    })
    
    # Group by query to ensure each batch contains products for the same query
    return list_wise_data

def prepare_ranking_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for ranking training with 1:2:1 ratio of positive:random:not-clicked samples
    """
    # Get positive samples (clicked)
    positive_samples = df[df['esci_label'] == 'E'].copy()
    
    if len(positive_samples) == 0:
        raise ValueError("No positive samples found in the dataset")
    
    # Calculate number of negative samples needed
    n_random = len(positive_samples) * 2  # 2x positive samples
    n_not_clicked = len(positive_samples)  # 1x positive samples
    
    # Sample random negative products
    random_negative = sample_negative_products(
        positive_samples,
        df,
        n_random
    )
    
    # Sample not-clicked products
    not_clicked = df[df['esci_label'] == 'I'].copy()
    if len(not_clicked) == 0:
        print("Warning: No not-clicked products found, using random negative samples instead")
        not_clicked = random_negative.sample(n=n_not_clicked, replace=True)
    else:
        # Remove duplicates from not_clicked
        not_clicked = not_clicked.drop_duplicates(subset=['product_id'])
        
        # Sample not-clicked products
        if len(not_clicked) < n_not_clicked:
            print(f"Warning: Not enough unique not-clicked samples, sampling with replacement to get {n_not_clicked} samples")
            not_clicked = not_clicked.sample(
                n=n_not_clicked,
                replace=True,
                weights='sampling_weight'
            )
        else:
            not_clicked = not_clicked.sample(
                n=n_not_clicked,
                replace=False,
                weights='sampling_weight'
            )
    
    # Combine all samples
    return pd.concat([
        positive_samples,
        random_negative,
        not_clicked
    ]).reset_index(drop=True)

def load_data(usage: str = "recall", training_method: str = "point_wise") -> Union[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], 
                                                                                 Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                                                                                 Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """
    Load and prepare data for either recall or ranking tasks
    
    Args:
        usage: Either "recall" or "ranking" to specify which data to return
        training_method: One of "point_wise", "pair_wise", or "list_wise"
        
    Returns:
        For point_wise:
            train_inputs, train_labels, val_inputs, val_labels
        For pair_wise:
            (train_q, train_p_pos, train_p_neg), train_labels, (val_q, val_p_pos, val_p_neg), val_labels
        For list_wise:
            (train_q, train_p, train_r), train_labels, (val_q, val_p, val_r), val_labels
    """
    # Try to load the processed dataset, if not available, prepare and save it
    try:
        print("Loading processed dataset...")
        df_dataset = pd.read_csv('versions/1/dataset_150k_processed.csv')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Processed dataset not found. Preparing new dataset...")
        df_dataset = prepare_and_save_dataset()
    
    # Split into train and test
    train_data = df_dataset[df_dataset['split'] != 'test']
    test_data = df_dataset[df_dataset['split'] == 'test']
    
    # Define column names
    query_tower_cols = [f'q{i}' for i in range(32)]
    product_tower_cols = [f'p{i}' for i in range(160)]
    
    if usage == "recall":
        if training_method == "point_wise":
            # Prepare point-wise data
            train_data, _ = prepare_point_wise_data(train_data)
            test_data, _ = prepare_point_wise_data(test_data)
            
            # Prepare training data
            train_inputs = [
                np.array(train_data[query_tower_cols]),
                np.array(train_data[product_tower_cols])
            ]
            train_labels = np.array(train_data['esci_label'] == 'E').astype('float32')
            
            # Prepare validation data
            val_inputs = [
                np.array(test_data[query_tower_cols]),
                np.array(test_data[product_tower_cols])
            ]
            val_labels = np.array(test_data['esci_label'] == 'E').astype('float32')
            
        elif training_method == "pair_wise":
            # Prepare pair-wise data
            train_pairs = prepare_pair_wise_data(train_data)
            test_pairs = prepare_pair_wise_data(test_data)
            
            # Prepare training data
            train_inputs = [
                np.array(train_pairs[[f'q{i}' for i in range(32)]]),
                np.array(train_pairs[[f'p{i}_positive' for i in range(160)]]),
                np.array(train_pairs[[f'p{i}_negative' for i in range(160)]])
            ]
            train_labels = np.ones(len(train_pairs))
            
            # Prepare validation data
            val_inputs = [
                np.array(test_pairs[[f'q{i}' for i in range(32)]]),
                np.array(test_pairs[[f'p{i}_positive' for i in range(160)]]),
                np.array(test_pairs[[f'p{i}_negative' for i in range(160)]])
            ]
            val_labels = np.ones(len(test_pairs))
            
        elif training_method == "list_wise":
            # Prepare list-wise data
            train_data = prepare_list_wise_data(train_data)
            test_data = prepare_list_wise_data(test_data)
            
            # Prepare training data
            train_inputs = [
                np.array(train_data[query_tower_cols]),
                np.array(train_data[product_tower_cols]),
                np.array(train_data['reward'])
            ]
            train_labels = np.array(train_data['esci_label'] == 'E').astype('float32')
            
            # Prepare validation data
            val_inputs = [
                np.array(test_data[query_tower_cols]),
                np.array(test_data[product_tower_cols]),
                np.array(test_data['reward'])
            ]
            val_labels = np.array(test_data['esci_label'] == 'E').astype('float32')
            
        else:
            raise ValueError("training_method must be one of: point_wise, pair_wise, list_wise")
            
    elif usage == "ranking":
        # For ranking, we only use point-wise training
        if training_method != "point_wise":
            print("Warning: Ranking only supports point-wise training. Switching to point-wise.")
            training_method = "point_wise"
        
        # Prepare ranking data with 1:2:1 ratio
        ranking_train = prepare_ranking_data(train_data)
        ranking_test = prepare_ranking_data(test_data)
        
        # Prepare training data
        train_inputs = [
            np.array(ranking_train[query_tower_cols]),
            np.array(ranking_train[product_tower_cols])
        ]
        train_labels = np.array(ranking_train['esci_label'] == 'E').astype('float32')
        
        # Prepare validation data
        val_inputs = [
            np.array(ranking_test[query_tower_cols]),
            np.array(ranking_test[product_tower_cols])
        ]
        val_labels = np.array(ranking_test['esci_label'] == 'E').astype('float32')
    
    else:
        raise ValueError("usage must be either 'recall' or 'ranking'")
    
    return train_inputs, train_labels, val_inputs, val_labels

if __name__ == "__main__":
    # Test point-wise data loading
    # point_wise_train_inputs, point_wise_train_labels, point_wise_val_inputs, point_wise_val_labels = load_data(usage="recall", training_method="point_wise")
    # print("Point-wise data shapes:")
    # print(f"Train inputs: {[x.shape for x in point_wise_train_inputs]}")
    # print(f"Train labels: {point_wise_train_labels.shape}")
    # print(f"Val inputs: {[x.shape for x in point_wise_val_inputs]}")
    # print(f"Val labels: {point_wise_val_labels.shape}")
    
    # Test pair-wise data loading
    pair_wise_train_inputs, pair_wise_train_labels, pair_wise_val_inputs, pair_wise_val_labels = load_data(usage="recall", training_method="pair_wise")
    print("\nPair-wise data shapes:")
    print(f"Train inputs: {[x.shape for x in pair_wise_train_inputs]}")
    print(f"Train labels: {pair_wise_train_labels.shape}")
    print(f"Val inputs: {[x.shape for x in pair_wise_val_inputs]}")
    print(f"Val labels: {pair_wise_val_labels.shape}")
    
    # Test list-wise data loading
    list_wise_train_inputs, list_wise_train_labels, list_wise_val_inputs, list_wise_val_labels = load_data(usage="recall", training_method="list_wise")
    print("\nList-wise data shapes:")
    print(f"Train inputs: {[x.shape for x in list_wise_train_inputs]}")
    print(f"Train labels: {list_wise_train_labels.shape}")
    print(f"Val inputs: {[x.shape for x in list_wise_val_inputs]}")
    print(f"Val labels: {list_wise_val_labels.shape}")

