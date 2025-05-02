import pandas as pd
import numpy as np

def load_data():

    # load product embedding and title
    df_product_embedding = pd.merge(
        # product_id,p0-p159,
        pd.read_csv('versions/1/product_150k.csv'), 
        # product_id,product_title,product_description,product_bullet_point,product_brand,product_color,product_locale
        pd.read_parquet('versions/1/shopping_queries_dataset_products.parquet')[['product_id','product_title']].drop_duplicates(),
        on = ['product_id']
    ).reset_index(drop=True)

    df_product_embedding['pid'] = range(0, df_product_embedding.shape[0])
    # load query embedding and title
    df_dataset = pd.merge(
        # query,product_id,esci_label,split, q0-q31,p0-p159,
        pd.read_csv('versions/1/dataset_150k.csv'), 
        df_product_embedding[['product_id','product_title']].drop_duplicates(), on = ['product_id']
        )

    # Convert 'esci_label' column to binary labels
    df_dataset['binary_label'] = df_dataset['esci_label'].apply(lambda x: 1 if x == 'E' else 0)

    # Split the dataset into training and validation
    train_data = df_dataset[df_dataset['split'] != 'test']
    val_data = df_dataset[df_dataset['split'] == 'test']

    train_labels = np.array(train_data['binary_label'])
    val_labels = np.array(val_data['binary_label'])

    train_labels = train_labels.astype('float32')
    val_labels = val_labels.astype('float32')

    query_tower_cols = ['q{}'.format(i) for i in range(32)]

    product_tower_cols = ['p{}'.format(i) for i in range(160)]

    # Prepare input data for training and validation
    train_inputs = [
        np.array(train_data[query_tower_cols]),
        np.array(train_data[product_tower_cols])
        
    ]

    val_inputs = [
        np.array(val_data[query_tower_cols]),
        np.array(val_data[product_tower_cols])
    ]

    return train_inputs, train_labels, val_inputs, val_labels

