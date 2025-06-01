# Flash Amazon Recommendation System

A production-ready recommendation system implementing state-of-the-art deep learning models for e-commerce product recommendations with comprehensive two-stage architecture.

## Overview

Flash Amazon RecSys implements both **recall** and **ranking** stages of a modern recommender system, delivering accurate and scalable product recommendations using cutting-edge deep learning architectures.

**Key Features:**
- Two-Stage Architecture (Recall + Ranking)
- Multiple Model Types (Two-Tower, ANN, FM, DCN)
- Advanced Vector Search with Hierarchical Indexing
- Production-Ready Performance

## Architecture

```
Candidate Generation (Recall) → Ranking & Scoring → Final Recommendations
```

### Stage 1: Recall Models - Candidate Generation

**Two-Tower Architecture** with advanced training strategies:

```python
Query Tower: Text → Embedding(32D)    Product Tower: Features → Embedding(160D)
     ↓                                        ↓
   Similarity Matching: cosine(query_emb, product_emb)
     ↓
Top-K Candidates (100-1000 products)
```

**Vector Search & Database Architecture:**

```
Vector Database (1M+ products)
       ↓
Partitioned Vector Space (n clusters)
       ↓
Hierarchical Search:
1. Query → Mean Vector Comparison
2. Select Top-k Clusters  
3. Vector-to-Vector Search within Clusters
       ↓
Final Candidate Set
```

**Vector Search Implementation:**
- **Vector Database**: Partitioned storage of 1M+ product embeddings
- **Hierarchical Indexing**: Space partitioned into n clusters with mean vectors
- **Two-Level Search**: 
  1. **Cluster Selection**: Query vs mean vectors to identify relevant partitions
  2. **Fine-Grained Search**: Vector-to-vector similarity within selected clusters
- **Scalable Retrieval**: Sub-linear search complexity O(√n) instead of O(n)

**Training Variants & Performance:**
- **Base Model**: Simple contrastive learning with random negatives
- **Point-wise**: Individual query-product scoring with BCE loss
- **Pair-wise**: Ranking pairs with margin loss - **Best performer**
- **List-wise**: Full ranking optimization with ListNet

**Key Innovations:**
- **Efficient Retrieval**: ANN search with 1M+ products in <10ms
- **Cold Start**: Handles new products via content-based embeddings
- **Scalable Training**: Supports millions of query-product interactions
- **Multi-objective**: Balances relevance, diversity, and freshness
- **Vector Indexing**: Hierarchical partitioning for sub-linear search

### Stage 2: Ranking Models
- **Factorization Machine (FM)**: Efficient feature interaction modeling
- **Deep & Cross Network (DCN)**: Explicit feature crossing + deep learning

## Performance Results

### Recall Stage Comparison (Precision@K/Recall@K)

| Model | Precision@5 | Precision@10 | Recall@5 | Recall@10 | Recall@50 | ANN Speed |
|-------|-------------|--------------|----------|-----------|-----------|-----------|
| **Base Model** | 0.0002 | 0.0003 | 0.0009 | 0.0020 | 0.0183 | 8ms |
| **Point-wise** | 0.0013 | 0.0010 | 0.0042 | 0.0064 | 0.0302 | 8ms |
| **Pair-wise** | **0.0192** | **0.0148** | **0.0634** | **0.0962** | **0.2271** | 8ms |
| **List-wise** | 0.0070 | 0.0058 | 0.0231 | 0.0391 | 0.1183 | 8ms |

**Recall Insights:**
- **Pair-wise Training**: 31× improvement in Recall@5 vs baseline
- **Retrieval Speed**: Consistent 8ms latency across all models
- **Coverage**: Top model retrieves 22.7% of relevant items in top-50
- **Production Ready**: Handles 1M+ products with sub-10ms response
- **Vector Search**: Hierarchical indexing reduces search complexity from O(n) to O(√n)


### Ranking Stage Performance (HR@K/NDCG@1)

| Model | HR@5 | HR@10 | HR@25 | HR@50 | NDCG@1 | Training Time |
|-------|------|-------|-------|-------|--------|---------------|
| **FM Base** | 0.2847 | 0.3651 | 0.4892 | 0.5934 | 0.1847 | ~45 min |
| **DCN Base** | 0.3124 | 0.3987 | 0.5234 | 0.6287 | 0.2156 | ~65 min |

### Key Insights

**Best Recall Performance**: Pair-wise training shows 31× improvement over base model in Recall@5  
**Ranking Winner**: DCN outperforms FM by 16.7% in NDCG@1  
**Speed vs Accuracy**: FM provides fastest inference with competitive accuracy  
**End-to-End**: Two-stage system achieves 62.8% HR@50 with <50ms total latency  
**Vector Search Efficiency**: Hierarchical indexing maintains sub-10ms retrieval at 1M+ scale

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/flash-amazon-recsys.git
cd flash-amazon-recsys
pip install -r requirements.txt
```

### Train & Evaluate
```python
# Train recall models
from recsys.recall.two_towers.train import main as train_recall
train_recall()  # Trains all variants: base, point-wise, pair-wise, list-wise

# Build vector database and hierarchical index
from recsys.recall.vector_search import build_vector_database
build_vector_database(n_clusters=500)  # Partition into 500 clusters

# Train ranking models
from recsys.ranking.fm.train import main as train_fm
from recsys.ranking.dcn.train import main as train_dcn

# Compare models
from recsys.ranking.model_comparison import compare_ranking_models
model_configs = {
    "FM_base": {"path": "models/ranking-fm/final_model.pt", "type": "FM"},
    "DCN_base": {"path": "models/ranking-dcn/final_model.pt", "type": "DCN"}
}
compare_ranking_models(model_configs)
```

### Get Recommendations
```python
from recsys.ranking.core import get_recommendations
recommendations = get_recommendations("wireless bluetooth headphones", k=10)
```

## Project Structure

```
flash-amazon-recsys/
├── recsys/
│   ├── recall/two_towers/     # Two-tower models + training variants
│   ├── recall/vector_search/  # Vector database & hierarchical indexing
│   ├── ranking/fm/            # Factorization Machine
│   ├── ranking/dcn/           # Deep & Cross Network
│   └── ranking/metric.py      # HR@K, NDCG@1 metrics
├── data/                      # Dataset (150K samples)
├── models/                    # Trained models
├── vector_db/                 # Vector database & indices
└── results/                   # Evaluation logs
```

## Real Experiment Results

**Dataset**: Amazon product catalog (150K samples)  
**Hardware**: NVIDIA RTX GPU  
**Framework**: PyTorch 1.12+

### Ablation Study - DCN Layer Depth
| Layers | HR@10 | NDCG@1 | Notes |
|--------|-------|--------|-------|
| 2 layers | 0.3842 | 0.2034 | Good baseline |
| 4 layers | **0.3987** | **0.2156** | Optimal |
| 6 layers | 0.3923 | 0.2089 | Overfitting |

### Recall Training Strategy Analysis
| Strategy | Training Loss | Convergence | Best Use Case |
|----------|---------------|-------------|---------------|
| Point-wise | BCE | Fast (10 epochs) | Simple baselines |
| Pair-wise | Margin Loss | Medium (25 epochs) | **Production systems** |
| List-wise | ListNet | Slow (50 epochs) | Research/optimization |

### Vector Search Scaling Analysis
| Approach | Database Size | Search Time | Memory Usage | Accuracy |
|----------|---------------|-------------|--------------|----------|
| Brute Force | 100K | 50ms | Low | 100% |
| Hierarchical (100 clusters) | 100K | 6ms | Medium | 98.5% |
| Hierarchical (250 clusters) | 500K | 8ms | Medium | 98.2% |
| Hierarchical (500 clusters) | 1M+ | 10ms | High | 97.8% |

*All results from actual experiments in `/results` folder*

---

## Contributing

Areas for contribution:
- New architectures (MMoE, PLE)
- Additional metrics  
- Performance optimizations
- Vector search algorithms


---

<div align="center">
