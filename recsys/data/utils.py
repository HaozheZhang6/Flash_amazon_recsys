import pandas as pd

def find_products_by_id(product_parquet_path, product_ids):
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(product_parquet_path)
    # Filter the DataFrame to include only the rows with the specified product IDs
    filtered_df = df[df['product_id'].isin(product_ids)]
    # Return the filtered DataFrame
    return filtered_df