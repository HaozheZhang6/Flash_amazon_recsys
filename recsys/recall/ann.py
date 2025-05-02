from typing import override
from annoy import AnnoyIndex
import pandas as pd
import numpy as np

class AnnoyRecall:

    def __init__(self, query_tower_input_dim, product_tower_input_dim, top_k=20):
        self.query_tower_input_dim = query_tower_input_dim
        self.product_tower_input_dim = product_tower_input_dim
        self.top_k = top_k

    def build_tree(self, df_query_embedding, df_product_embedding):
        q = AnnoyIndex(self.query_tower_input_dim, 'euclidean')
        mp_query_dict = {}
        target_col = ['q{}'.format(x) for x in list(range(query_tower_input_dim))]
        for _, row in df_query_embedding.iterrows():
            mp_query_dict[row['qid']] = row['query']
            key = int(row['qid'])
            vec = list(row[target_col])
            q.add_item(key,vec)
        q.build(100) # 100 trees
        q.save('query.tree')
        p = AnnoyIndex(self.product_tower_input_dim, 'euclidean')
        mp_product_dict = {}
        for ix,row in df_product_embedding.iterrows():
            mp_product_dict[int(row['pid'])] = row['product_title']
            key = int(row['pid'])
            vec = list(row[['p{}'.format(x) for x in list(range(self.product_tower_input_dim))]])
            p.add_item(key,vec)
        p.build(100) # 100 trees
        p.save('product.tree')
    
    def recall(self, df_query_embedding, df_product_embedding):
        # check if the tree is built
        try:
            q = AnnoyIndex(self.query_tower_input_dim, 'euclidean')
            q.load('query.tree')
            p = AnnoyIndex(self.product_tower_input_dim, 'euclidean')
            p.load('product.tree')
        except:
            self.build_tree(df_query_embedding, df_product_embedding)
            q = AnnoyIndex(self.query_tower_input_dim, 'euclidean')
            q.load('query.tree')
            p = AnnoyIndex(self.product_tower_input_dim, 'euclidean')
            p.load('product.tree')
        
        q = AnnoyIndex(self.query_tower_input_dim, 'euclidean')
        q.load('query.tree')
        p = AnnoyIndex(self.product_tower_input_dim, 'euclidean')
        p.load('product.tree')
        top_k = self.top_k
        mat = []
        for ix,row in df_query_embedding.iterrows():
            item = row['query']
            mat.append([item] + [mp_query_dict[x] for x in q.get_nns_by_item(row['qid'], top_k+1)[1:]])
            if ix == 50:
                break
        cols = ['query_id']
        for i in range(top_k):
            cols += ['nearest_{}'.format(i+1)]
        print(cols)
        df_neighbors1 = pd.DataFrame(mat, columns = cols)
        display(df_neighbors1.head(50))
        top_k = self.top_k
        mat = []
        for ix,row in df_product_embedding.iterrows():
            item = row['product_title']
            mat.append([item] + [mp_product_dict[x] for x in p.get_nns_by_item(row['pid'], top_k+1)[1:]])
            if ix == 50:
                break
        cols = ['product_id']
        for i in range(top_k):
            cols += ['nearest_{}'.format(i+1)]
        print(cols)
        df_neighbors2 = pd.DataFrame(mat, columns = cols)
        display(df_neighbors2.head(50))
        return df_neighbors1, df_neighbors2

if __name__ == '__main__':
    df_product_embedding = pd.merge(
    pd.read_csv('versions/1/product_150k.csv'),
    pd.read_parquet('versions/1/shopping_queries_dataset_products.parquet')[['product_id','product_title']].drop_duplicates(),
    on = ['product_id']
    ).reset_index(drop=True)

    df_product_embedding['pid'] = range(0, df_product_embedding.shape[0])

    df_query_embedding = pd.read_csv('versions/1/query_150k.csv')
    df_query_embedding['qid'] = range(0, df_query_embedding.shape[0])
    query_tower_input_dim = 32
    product_tower_input_dim = (32*5)
    top_k = 20
    mat = []

    ann_recall = AnnoyRecall(query_tower_input_dim, product_tower_input_dim, top_k)