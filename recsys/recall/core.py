from recsys.recall.two_towers.eval import *
from recsys.recall.ann import *
from recsys.data.load_data import load_data
from recsys.data.utils import *

def main():
    '''
    Given an array of query, find the top k products, 
    set 1: using two tower model to calculate the similarity with all products and select the top k/2,
    set 2: using ann to find the top m products of each of the top k/2 products
    select the top k products from the two sets
    '''
    
    df_products, df_dataset_mini = load_data()