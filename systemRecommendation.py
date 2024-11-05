import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained RoBERTa model
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()

# Load the Amazon products dataset
amazon_data = pd.read_csv('/home/sagemaker-user/fairseq-Bert-Rec-SystemRecommendation/Datasets/testdata_smallsize.csv')

# Use name and main_category for item representation
items = amazon_data[['name', 'main_category']].apply(lambda x: f"{x['name']} - Category: {x['main_category']}", axis=1).tolist()

# Embed all items
item_embeddings = [roberta.extract_features(roberta.encode(item)).mean(dim=1).detach().numpy().flatten() for item in items]

# Compute cosine similarities among items
similarity_matrix = cosine_similarity(item_embeddings)

# Example: Using the first item to generate recommendations

input_item_index = np.random.randint(len(items))  # Change this index to test different items
recommended_indices = similarity_matrix[input_item_index].argsort()[-2:-5:-1]  # Get top 3 recommendations
recommended_items = [items[idx] for idx in recommended_indices]

# Display input data and output
print("Input item:", items[input_item_index])
print("Recommended items:", recommended_items)

# Metrics calculations
true_positive = amazon_data['name'].iloc[input_item_index]  # The actual item for the given index
hit_rate = int(recommended_items[0] == true_positive)

# Mean Reciprocal Rank (MRR)
mrr = 1 / (recommended_indices[0] + 1)  # Rank of the first relevant item

# Normalized Discounted Cumulative Gain (NDCG)
k = 3  # Number of top recommendations
dcg = sum(1 / (idx + 1) for idx in range(k) if idx < len(recommended_items) and recommended_items[idx] == true_positive)
idcg = 1 / (1 + 1)  # Ideal DCG for a single relevant item
ndcg = dcg / idcg if idcg > 0 else 0

print("\n Metrics:")
print("Hit Rate:", hit_rate)
print("Mean Reciprocal Rank (MRR):", mrr)
print("Normalized Discounted Cumulative Gain (NDCG):", ndcg)

# Display input data and output
print("\n Input item:", items[input_item_index])
print("Recommended items:", recommended_items)
