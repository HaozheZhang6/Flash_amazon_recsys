from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import glob

def reshape_array(input_array, d):
        k, _ = input_array.shape
        new_array = np.zeros((k, d))
        
        for i in range(k):
            for j in range(d):
                start_idx = j * (768 // d)
                end_idx = (j + 1) * (768 // d) if j < (d - 1) else 768
                chunk = input_array[i, start_idx:end_idx]
                new_array[i, j] = np.mean(chunk)
        
        return new_array

def convert_embedding(product_parquet='versions/1/shopping_queries_dataset_products.parquet', query_csv='versions/1/Dataset.csv'):
    
    # Load a pre-trained model (you can choose from various models like BERT, RoBERTa, etc.)
    model = SentenceTransformer('distilbert-base-uncased')
    step = 2000
    product_dim = 32
    query_dim = 32
    df_query_product_match = pd.read_csv(query_csv)
    sample_size = 2000
    df_query_product_match_mini = df_query_product_match.sample(sample_size, random_state = 42).reset_index(drop=True)

    df_products = pd.read_parquet(product_parquet)

    cols = ['product_title', 'product_description', 'product_bullet_point','product_brand','product_color','product_id']
    df_products = pd.merge(df_products[cols].drop_duplicates(), df_query_product_match_mini[['product_id']].drop_duplicates() ,on = ['product_id'])

    df_products['product_title'] = df_products['product_title'].apply(lambda x : str(x).lower() if pd.notna(x) else '')
    df_products['product_description'] = df_products['product_description'].apply(lambda x : str(x).lower() if pd.notna(x) else '')
    df_products['product_bullet_point'] = df_products['product_bullet_point'].apply(lambda x : str(x).lower() if pd.notna(x) else '')
    df_products['product_brand'] = df_products['product_brand'].apply(lambda x : str(x).lower() if pd.notna(x) else '')
    df_products['product_color'] = df_products['product_color'].apply(lambda x : str(x).lower() if pd.notna(x) else '')
    df_query_product_match_mini['query'] = df_query_product_match_mini['query'].apply(lambda x : str(x).lower() if pd.notna(x) else '')


    cols = ['q' + str(x) for x in list(range(0, query_dim))] + ['query']
    cnt = 0
    queries = list(df_query_product_match_mini['query'].unique())
    for i in range(0,len(queries),step):
        
        cnt += 1
        # Define a list of sentences
        sentences = list(queries)[i:i+step]

        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = sentence_embeddings.cpu()
        sentence_embeddings = reshape_array(np.array(sentence_embeddings), query_dim)
        
        df_tmp = pd.DataFrame(np.concatenate((sentence_embeddings, np.array(sentences).reshape(-1,1)), axis=1))
        
        df_tmp.columns = cols
        
        df_tmp.to_csv(
            f'query_{cnt}.csv', header = True, index = False)
    
    lst_product_title = list(df_products['product_title'])
    lst_product_description = list(df_products['product_description'])
    lst_product_bullet_point = list(df_products['product_bullet_point'])
    lst_product_brand = list(df_products['product_brand'])
    lst_product_color = list(df_products['product_color'])
    lst_product_id = list(df_products['product_id'])


    cols = ['p' + str(x) for x in list(range(0, product_dim * 5))] + ['product_id']
    def find_embeddings(lst_product_title,i, product_dim):
        # Define a list of sentences
        sentences = list(lst_product_title)[i:i+step]
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = sentence_embeddings.cpu()
        return reshape_array(np.array(sentence_embeddings), product_dim)
    cnt = 0
    for i in range(0,len(lst_product_title),step):
        cnt += 1
        product_title_embed = find_embeddings(lst_product_title, i, product_dim)
        product_description_embed = find_embeddings(lst_product_description, i, product_dim)
        product_bullet_point_embed = find_embeddings(lst_product_bullet_point, i, product_dim)
        product_brand_embed = find_embeddings(lst_product_brand, i, product_dim)
        product_color_embed = find_embeddings(lst_product_color, i, product_dim)
        
        # Concatenate arrays column-wise and reshape lst_product_id to (x, 1)
        df_tmp = pd.DataFrame(np.concatenate((
            product_title_embed,
            product_description_embed,
            product_bullet_point_embed,
            product_brand_embed,
            product_color_embed,
            np.array(lst_product_id[i:i + step]).reshape(-1, 1)
        ), axis=1))
        
        df_tmp.columns = cols
        df_tmp.to_csv(f'product_{cnt}.csv', header = True, index = False)

    def connect_csv(file_list, target_file):
        dfs = []
        for file in file_list:
            df = pd.read_csv(file)
            dfs.append(df)
        concatenated_df = pd.concat(dfs, ignore_index=True)
        concatenated_df.to_csv(target_file, index=False)

    file_list = glob.glob('product_*.csv')
    connect_csv(file_list, 'product.csv') # product_id+ embedding
    file_list = glob.glob('query_*.csv')
    connect_csv(file_list, 'query.csv') # query + embedding


def convert_query(input_query):
    'convert an input query list to embedding vector with default dim = 32'
    model = SentenceTransformer('distilbert-base-uncased')
    query_dim = 32
    # lower case
    input_query = [str(x).lower() for x in input_query]
    sentence_embeddings = model.encode(input_query, convert_to_tensor=True)
    sentence_embeddings = sentence_embeddings.cpu()
    sentence_embeddings = reshape_array(np.array(sentence_embeddings), query_dim)
    return sentence_embeddings



if __name__ == '__main__':
    convert_embedding()