import numpy as np
from pyspark.sql import functions as F
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def dynamic_triplet_mining(
        anchor_df,
        pos_neg_df,
        join_keys,
        anchor_encoder, 
        pos_neg_encoder, 
        stage="random", 
        sample_size=100000,
        semi_hard_ratio=0.3, 
        hard_ratio=0.5, 
        top_k=50000, 
        chunk_size=5000):
    
    if isinstance(join_keys, str):
        join_keys = [join_keys]

    anchor_df = anchor_df.alias("anchor")
    pos_neg_df = pos_neg_df.alias("pos_neg")

    # --- Positive pairs ---
    match_df = anchor_df \
        .join(pos_neg_df, on=join_keys, how="inner") \
        .select(
            F.col("anchor.features").alias("anchor_features"), 
            F.col("pos_neg.features").alias("positive_features")
        )

    positives_pd = match_df.limit(sample_size).toPandas()
    anchor_features = positives_pd["anchor_features"].apply(pd.Series)
    positive_features = positives_pd["positive_features"].apply(pd.Series)

    # --- Candidate negatives ---
    negatives_df = pos_neg_df.orderBy(F.rand()).limit(sample_size * 3)
    negatives_pd = negatives_df.select("features").toPandas()
    neg_features = negatives_pd["features"].apply(pd.Series)

    # --- Compute embeddings ---
    anchor_embeds = anchor_encoder.predict(anchor_features, batch_size=512, verbose=0)
    neg_embeds = pos_neg_encoder.predict(neg_features, batch_size=512, verbose=0)

    # --- Compute similarity matrix in chunks ---
    similarity_matrix = np.zeros((len(anchor_embeds), len(neg_embeds)), dtype=np.float32)
    for start in range(0, len(anchor_embeds), chunk_size):
        end = min(start + chunk_size, len(anchor_embeds))
        sim_chunk = cosine_similarity(anchor_embeds[start:end], neg_embeds)
        similarity_matrix[start:end] = sim_chunk

    # --- Select negatives based on stage ---
    if stage == "random":
        # All negatives are random
        selected_neg_indices = np.random.choice(len(neg_embeds), size=sample_size, replace=False)

    elif stage == "semi-hard":
        semi_hard_count = int(sample_size * semi_hard_ratio)
        random_count = sample_size - semi_hard_count

        # Semi-hard negatives
        semi_hard_indices = []
        for i in range(len(anchor_embeds)):
            mask = (similarity_matrix[i] > 0.3) & (similarity_matrix[i] < 0.7)
            candidates = np.where(mask)[0]
            if len(candidates) > 0:
                semi_hard_indices.append(np.random.choice(candidates))
            else:
                semi_hard_indices.append(np.random.randint(len(neg_embeds)))
        semi_hard_indices = np.random.choice(semi_hard_indices, size=semi_hard_count, replace=False)

        # Random negatives
        random_indices = np.random.choice(len(neg_embeds), size=random_count, replace=False)

        selected_neg_indices = np.concatenate([semi_hard_indices, random_indices])

    elif stage == "hard":
        hard_count = int(sample_size * hard_ratio)
        random_count = sample_size - hard_count

        # Hard negatives
        hard_candidates = []
        top_k = min(top_k, similarity_matrix.shape[1])
        for i in range(len(anchor_embeds)):
            kth = top_k - 1
            top_indices = np.argpartition(-similarity_matrix[i], kth)[:top_k]
            hard_candidates.append(np.random.choice(top_indices))
        hard_indices = np.random.choice(hard_candidates, size=hard_count, replace=False)

        # Random negatives
        random_indices = np.random.choice(len(neg_embeds), size=random_count, replace=False)

        selected_neg_indices = np.concatenate([hard_indices, random_indices])

    else:
        raise ValueError("Invalid stage")

    # --- Return triplets ---
    selected_neg_features = neg_features.iloc[selected_neg_indices].reset_index(drop=True)
    return anchor_features, positive_features, selected_neg_features