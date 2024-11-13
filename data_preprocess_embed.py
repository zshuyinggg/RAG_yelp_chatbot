#%%
import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

#%%
import gc
gc.collect()
#%%
business_data = load_data('./data/yelp_academic_dataset_business.json')
review_data = load_data('./data/yelp_academic_dataset_review.json')


#%%
review_df = pd.DataFrame(review_data)
business_df = pd.DataFrame(business_data)
Philadelphia_df = business_df[business_df['city'] == 'Philadelphia']
food_related_df = Philadelphia_df[Philadelphia_df['categories'].str.contains('food|restaurant|cafe|bistro|diner|eatery|bar|pub|grill|buffet|bakery|pizzeria|'
    'sushi|steakhouse|noodle|ramen|bbq|bbq|barbecue|tavern|'
    'coffee|tea|dessert|pastry|pasta|seafood|'
    'brunch|breakfast|lunch|dinner|takeout|delivery|sandwich|burger|'
    'wine|beer|brewery|gelato|smoothie|juice', case=False, na=False)]
review_df = review_df[review_df['business_id'].isin(Philadelphia_df['business_id'])]

#%%


review_df_grouped = review_df.groupby('business_id')['text'].apply(lambda x: ' '.join(x)).reset_index()
merge_df = pd.merge(business_df, review_df_grouped, on='business_id', how='left')
merge_df.rename(columns={'text': 'all_reviews'}, inplace=True)
merge_df = merge_df[['business_id', 'name', 'address', 'attributes', 'categories', 'all_reviews']]

merge_df.to_csv('./data/merge_businessinfo_reviews_philadelphia.csv', index=False)
#%%
merge_df['full_context'] = merge_df['name'] + ' | ' + merge_df['address'] + ' | ' + merge_df['attributes'].astype(str) + ' | ' + merge_df['categories'] + ' | ' + merge_df['all_reviews']


model_minilm = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
model_mpnet = SentenceTransformer('all-mpnet-base-v2', device='cuda')

embeddings_minilm = model_minilm.encode(merge_df['full_context'].tolist(), show_progress_bar=True)
embeddings_mpnet = model_mpnet.encode(merge_df['full_context'].tolist(), show_progress_bar=True)

index_minilm = faiss.IndexFlatL2(384)  # MiniLM 384-dimensional embeddings
index_mpnet = faiss.IndexFlatL2(768)   # MPNet 768-dimensional embeddings

index_minilm.add(np.array(embeddings_minilm))
index_mpnet.add(np.array(embeddings_mpnet))

faiss.write_index(index_minilm, '/data/embedding/faiss_index_minilm_philadelphia.index')
faiss.write_index(index_mpnet, '/data/embedding/faiss_index_mpnet_philadelphia.index')


merge_df['embeddings_minilm'] = list(embeddings_minilm)
merge_df['embeddings_mpnet'] = list(embeddings_mpnet)

merge_df.to_csv('/content/drive/MyDrive/cs646/final_rag/data_processing/yelp_filtered/merge_businessinfo_reviews_with_embeddings.csv', index=False)