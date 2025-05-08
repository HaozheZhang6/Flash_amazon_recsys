import pandas as pd
import numpy as np
from typing import Tuple, List

def calculate_sampling_weights(df: pd.DataFrame) -> pd.Series:
    """
    Calculate sampling weights based on click counts raised to power 0.75
    Returns a Series indexed by product_id
    """
    click_counts = df.groupby('product_id').size()
    weights = click_counts ** 0.75
    return weights / weights.sum()

def sample_negative_products(
    positive_products: pd.DataFrame,
    all_products: pd.DataFrame,
    n_samples: int,
    weights: pd.Series
) -> pd.DataFrame:
    """
    Sample negative products based on weights
    """
    # Get available products (not in positive samples)
    available_products = all_products[~all_products['product_id'].isin(positive_products['product_id'])]
    
    if len(available_products) == 0:
        # If no available products, sample from all products with replacement
        print("Warning: No available negative products, sampling with replacement")
        sampled_product_ids = np.random.choice(
            all_products['product_id'].unique(),
            size=n_samples,
            replace=True
        )
        sampled_products = all_products[all_products['product_id'].isin(sampled_product_ids)]
    else:
        # Get weights for available products
        available_weights = weights[weights.index.isin(available_products['product_id'])]
        
        if len(available_weights) == 0:
            # If no weights available, use uniform sampling
            print("Warning: No weights available for negative sampling, using uniform sampling")
            sampled_products = available_products.sample(
                n=min(n_samples, len(available_products)),
                replace=len(available_products) < n_samples
            )
        else:
            # Normalize weights for available products
            available_weights = available_weights / available_weights.sum()
            
            # Sample products
            sampled_product_ids = np.random.choice(
                available_weights.index,
                size=min(n_samples, len(available_weights)),
                p=available_weights.values,
                replace=len(available_weights) < n_samples
            )
            
            # Get the full product data for sampled IDs
            sampled_products = available_products[available_products['product_id'].isin(sampled_product_ids)]
    
    # If we still don't have enough samples, sample with replacement
    if len(sampled_products) < n_samples:
        print(f"Warning: Not enough unique negative samples, sampling with replacement to get {n_samples} samples")
        additional_samples = n_samples - len(sampled_products)
        additional_products = sampled_products.sample(n=additional_samples, replace=True)
        sampled_products = pd.concat([sampled_products, additional_products])
    
    return sampled_products

def prepare_recall_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for recall training with 1:3 positive to negative ratio
    """
    # Get positive samples (clicked)
    positive_samples = df[df['esci_label'] == 'E'].copy()
    
    if len(positive_samples) == 0:
        raise ValueError("No positive samples found in the dataset")
    
    # Calculate sampling weights
    weights = calculate_sampling_weights(positive_samples)
    
    # Sample negative products (3 times the number of positive samples)
    negative_samples = sample_negative_products(
        positive_samples,
        df,
        len(positive_samples) * 3,
        weights
    )
    
    # Combine positive and negative samples
    recall_data = pd.concat([
        positive_samples,
        negative_samples
    ]).reset_index(drop=True)
    
    return recall_data, weights

def prepare_ranking_data(
    df: pd.DataFrame,
    weights: pd.Series
) -> pd.DataFrame:
    """
    Prepare data for ranking training with 1:3 positive to negative ratio
    and 1:1 random negative to not-clicked ratio
    """
    # Get positive samples (clicked)
    positive_samples = df[df['esci_label'] == 'E'].copy()
    
    if len(positive_samples) == 0:
        raise ValueError("No positive samples found in the dataset")
    
    # Calculate number of negative samples needed
    n_negative = len(positive_samples) * 3
    
    # Sample random negative products (half of negative samples)
    random_negative = sample_negative_products(
        positive_samples,
        df,
        n_negative // 2,
        weights
    )
    
    # Sample not-clicked products (half of negative samples)
    not_clicked = df[df['esci_label'] == 'I'].copy()
    if len(not_clicked) == 0:
        print("Warning: No not-clicked products found, using random negative samples instead")
        not_clicked = random_negative
    else:
        # Remove duplicates from not_clicked to avoid reindexing issues
        not_clicked = not_clicked.drop_duplicates(subset=['product_id'])
        
        # Get weights for not_clicked products
        not_clicked_weights = weights[weights.index.isin(not_clicked['product_id'])]
        
        if len(not_clicked_weights) == 0:
            print("Warning: No weights available for not-clicked sampling, using uniform sampling")
            not_clicked = not_clicked.sample(
                n=min(n_negative // 2, len(not_clicked)),
                replace=len(not_clicked) < n_negative // 2
            )
        else:
            # Normalize weights
            not_clicked_weights = not_clicked_weights / not_clicked_weights.sum()
            
            # Sample products
            sampled_product_ids = np.random.choice(
                not_clicked_weights.index,
                size=min(n_negative // 2, len(not_clicked_weights)),
                p=not_clicked_weights.values,
                replace=len(not_clicked_weights) < n_negative // 2
            )
            
            # Get the full product data for sampled IDs
            not_clicked = not_clicked[not_clicked['product_id'].isin(sampled_product_ids)]
    
    # If we still don't have enough not-clicked samples, sample with replacement
    if len(not_clicked) < n_negative // 2:
        print(f"Warning: Not enough unique not-clicked samples, sampling with replacement to get {n_negative // 2} samples")
        additional_samples = n_negative // 2 - len(not_clicked)
        additional_products = not_clicked.sample(n=additional_samples, replace=True)
        not_clicked = pd.concat([not_clicked, additional_products])
    
    # Combine all samples
    ranking_data = pd.concat([
        positive_samples,
        random_negative,
        not_clicked
    ]).reset_index(drop=True)
    
    return ranking_data

def load_data(usage: str = "recall") -> Tuple[Tuple[np.ndarray, np.ndarray],
                                             Tuple[np.ndarray, np.ndarray]]:
    """
    Load and prepare data for either recall or ranking tasks
    
    Args:
        usage: Either "recall" or "ranking" to specify which data to return
        
    Returns:
        For recall (default):
            train_inputs, train_labels, val_inputs, val_labels
        For ranking:
            train_inputs, train_labels, val_inputs, val_labels
    """
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
    
    # Split into train and test
    train_data = df_dataset[df_dataset['split'] != 'test']
    test_data = df_dataset[df_dataset['split'] == 'test']
    
    # Define column names
    query_tower_cols = [f'q{i}' for i in range(32)]
    product_tower_cols = [f'p{i}' for i in range(160)]
    
    if usage == "recall":
        # Prepare recall data
        recall_train, train_weights = prepare_recall_data(train_data)
        recall_test, test_weights = prepare_recall_data(test_data)
        
        # Prepare training data
        train_inputs = [
            np.array(recall_train[query_tower_cols]),
            np.array(recall_train[product_tower_cols])
        ]
        train_labels = np.array(recall_train['esci_label'] == 'E').astype('float32')
        
        # Prepare validation data
        val_inputs = [
            np.array(recall_test[query_tower_cols]),
            np.array(recall_test[product_tower_cols])
        ]
        val_labels = np.array(recall_test['esci_label'] == 'E').astype('float32')
        
    elif usage == "ranking":
        # Prepare ranking data
        ranking_train = prepare_ranking_data(train_data, calculate_sampling_weights(train_data))
        ranking_test = prepare_ranking_data(test_data, calculate_sampling_weights(test_data))
        
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
    # Test recall data loading
    recall_train_inputs, recall_train_labels, recall_val_inputs, recall_val_labels = load_data(usage="recall")
    print("Recall data shapes:")
    print(f"Train inputs: {[x.shape for x in recall_train_inputs]}")
    print(f"Train labels: {recall_train_labels.shape}")
    print(f"Val inputs: {[x.shape for x in recall_val_inputs]}")
    print(f"Val labels: {recall_val_labels.shape}")
    
    # Test ranking data loading
    ranking_train_inputs, ranking_train_labels, ranking_val_inputs, ranking_val_labels = load_data(usage="ranking")
    print("\nRanking data shapes:")
    print(f"Train inputs: {[x.shape for x in ranking_train_inputs]}")
    print(f"Train labels: {ranking_train_labels.shape}")
    print(f"Val inputs: {[x.shape for x in ranking_val_inputs]}")
    print(f"Val labels: {ranking_val_labels.shape}")

