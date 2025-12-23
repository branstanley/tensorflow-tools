import numpy as np
from sklearn.metrics import roc_curve
from DualEncoderSimilarityModel.dynamic_triplet_mining import dynamic_triplet_mining
from LossFunctions.combined_triplet_contrastive_loss import combined_triplet_contrastive_loss as combined_loss
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import math
import tensorflow as tf

# Dynamic margin variable for loss
margin_var = tf.Variable(0.5, trainable=False)

def train_with_stagewise_margin(
        anchor_df,
        pos_neg_df,
        anchor_test_df,
        pos_neg_test_df,
        join_keys,
        triplet_model,
        anchor_encoder, 
        pos_neg_encoder, 
        epochs=20, 
        training_iterations=3, 
        sample_size=5000, 
        threshold=0.7):
    
    # Compile once with dynamic margin
    triplet_model.compile(optimizer='adam', loss=combined_loss())

    # Stage definitions
    stages = [
        ("random", 0.1, 0.3),
        ("semi-hard", 0.1, 0.3),
        ("hard", 0.1, 0.3)
    ]

    # Initial triplet mining for training
    anchor, positive, negative = dynamic_triplet_mining(
        anchor_df,
        pos_neg_df,
        join_keys,
        anchor_encoder, 
        pos_neg_encoder, 
        stage="random",
        sample_size=sample_size
    )

    # Calculate stage boundaries
    stage_epochs = epochs // len(stages)
    extra_epochs = epochs % len(stages)
    random_end = stage_epochs
    semi_hard_end = stage_epochs * 2
    hard_end = epochs

    final_best_threshold = None

    for epoch in range(1, epochs + 1):
        # Determine stage and margin
        if epoch <= random_end:
            stage_name, stage_min, stage_max = stages[0]
            progress = (epoch - 1) / max(random_end - 1, 1)
            current_margin = stage_min + progress * (stage_max - stage_min)
        elif epoch <= semi_hard_end:
            stage_name, stage_min, stage_max = stages[1]
            progress = (epoch - random_end - 1) / max(stage_epochs - 1, 1)
            current_margin = stage_min + progress * (stage_max - stage_min)
        else:
            stage_name, stage_min, stage_max = stages[2]
            hard_start = semi_hard_end + 1
            hard_progress = (epoch - hard_start) / max(hard_end - hard_start, 1)
            current_margin = stage_min + hard_progress * (stage_max - stage_min)

        margin_var.assign(current_margin)
        print(f"\nEpoch {epoch}/{epochs} - Stage: {stage_name}, Margin: {current_margin:.3f}")

        sample_weights = np.ones(len(anchor))

        # --- Training iterations ---
        for iteration in range(training_iterations):
            anchor_embeds = anchor_encoder.predict(anchor, batch_size=128, verbose=0)
            positive_embeds = pos_neg_encoder.predict(positive, batch_size=128, verbose=0)
            negative_embeds = pos_neg_encoder.predict(negative, batch_size=128, verbose=0)

            pos_sim = np.sum(anchor_embeds * positive_embeds, axis=1)
            neg_sim = np.sum(anchor_embeds * negative_embeds, axis=1)

            correct = (pos_sim > neg_sim + current_margin)
            confidence = np.abs(pos_sim - neg_sim)
            incorrect_count = np.sum(~correct)
            print(f"Iteration {iteration+1}/{training_iterations}: Incorrect = {incorrect_count}/{len(anchor)} ({incorrect_count/len(anchor)*100:.2f}%)")

            current_weights = np.where(correct, 1.1,
                                       np.where(pos_sim < neg_sim, 1.5 + 0.5 * confidence, 1.0 + 0.5 * confidence))
            alpha = 0.6
            sample_weights = alpha * current_weights + (1 - alpha) * sample_weights

            dummy_labels = np.zeros((len(anchor), 1))
            
            triplet_model.fit([anchor, positive, negative], dummy_labels,
                              batch_size=128, epochs=1, sample_weight=sample_weights, verbose=0)

        # Refresh triplets for next epoch
        anchor, positive, negative = dynamic_triplet_mining(
            anchor_df,
            pos_neg_df,
            join_keys,
            anchor_encoder, 
            pos_neg_encoder, 
            stage=stage_name,
            sample_size=sample_size
        )

        # --- Validation using test_df ---
        val_anchor, val_positive, val_negative = dynamic_triplet_mining(
            anchor_test_df,
            pos_neg_test_df,
            join_keys,
            anchor_encoder=anchor_encoder, 
            pos_neg_encoder=pos_neg_encoder, 
            stage="random",  # Always random for validation
            sample_size=sample_size
        )

        val_anchor_embeds = anchor_encoder.predict(val_anchor, batch_size=128, verbose=0)
        val_pos_embeds = pos_neg_encoder.predict(val_positive, batch_size=128, verbose=0)
        val_neg_embeds = pos_neg_encoder.predict(val_negative, batch_size=128, verbose=0)

        val_pos_sim = np.clip(np.sum(val_anchor_embeds * val_pos_embeds, axis=1), -1.0, 1.0)
        val_neg_sim = np.clip(np.sum(val_anchor_embeds * val_neg_embeds, axis=1), -1.0, 1.0)

        print(f"Validation Avg Positive Cosine Similarity: {np.mean(val_pos_sim):.4f}, Avg Negative Cosine Similarity: {np.mean(val_neg_sim):.4f}")

        all_scores = np.concatenate([val_pos_sim, val_neg_sim])
        y_true = np.concatenate([np.ones(len(val_anchor)), np.zeros(len(val_anchor))])

        fpr, tpr, thresholds = roc_curve(y_true, all_scores)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        final_best_threshold = best_threshold
        auc = roc_auc_score(y_true, all_scores)
        f1 = f1_score(y_true, all_scores > best_threshold)
        prec = precision_score(y_true, all_scores > best_threshold)
        rec = recall_score(y_true, all_scores > best_threshold)
        print(f"Validation ROC-AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, best_threshold: {best_threshold:.4f}")

    return final_best_threshold