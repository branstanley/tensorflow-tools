import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from DualEncoderSimilarityModel.add_projection_head import add_projection_head
from DualEncoderSimilarityModel.build_encoder import build_encoder
from DualEncoderSimilarityModel.train_with_stagewise_margin import train_with_stagewise_margin
from Tools.get_encoded_labels import get_encoded_labels

def train_encoder_model(df, input_shape=(650,), grouping_cols=["SomeGroup"], feature_col="features"):
    num_classes, classifier_input_values, classifier_input_labels = get_encoded_labels(df, grouping_cols=grouping_cols, feature_col=feature_col)

    features = np.array(classifier_input_values, dtype=np.float32)
    labels = np.array(classifier_input_labels, dtype=np.int32).reshape(-1)

    m,n = features.shape

    model = build_encoder(num_classes=num_classes, features=features, input_shape=input_shape)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels, 
        random_state=42
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

# Note: At this point I assume df1 and df2 are pre-defined Spark DataFrames for the two signals, and have already gone through feature engineering steps.
model_1 = train_encoder_model(df1, input_shape=(650,), grouping_cols="GroupA", feature_col="features")
model_2 = train_encoder_model(df2, input_shape=(616,), grouping_cols="GroupB", feature_col="features")

model_1.save("encoder_model_1")
model_2.save("encoder_model_2")

# After pretraining, freeze encoder
encoder_1 = tf.keras.Model(inputs=model_1.input, outputs=model_1.layers[-2].output)
encoder_1.trainable = False 
# After pretraining, freeze encoder
encoder_2 = tf.keras.Model(inputs=model_2.input, outputs=model_2.layers[-2].output)
encoder_2.trainable = False 

encoder_1 = add_projection_head(encoder_1, input_shape=(650,))
encoder_2 = add_projection_head(encoder_2, input_shape=(616,))

# Triplet inputs
anchor_input = tf.keras.layers.Input(shape=(650,), name="anchor")      # Anchor on signal 1
positive_input = tf.keras.layers.Input(shape=(616,), name="positive")  # Positive on signal 2
negative_input = tf.keras.layers.Input(shape=(616,), name="negative")  # Negative on signal 2

# Pass through encoders
anchor_embed = encoder_1(anchor_input)
positive_embed = encoder_2(positive_input)
negative_embed = encoder_2(negative_input)

# Concatenate embeddings for loss computation
triplet_output = tf.keras.layers.Concatenate(axis=1)([anchor_embed, positive_embed, negative_embed])

triplet_model = tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=triplet_output)

# Pre-split data for training and testing.  
threshold = train_with_stagewise_margin(training_guids_df, testing_guids_df, triplet_model, encoder_1, encoder_2,
                                epochs=10, training_iterations=5, sample_size=40000, threshold=0.7)

# Build final model then save it

encoder_1.trainable = False
encoder_2.trainable = False

input_1 = tf.keras.layers.Input(shape=(650,), name="source_1")
input_2 = tf.keras.layers.Input(shape=(616,), name="source_2")

# Get embeddings from trained encoders
embed_1 = encoder_1(input_1)     
embed_2 = encoder_2(input_2)

# Cosine similarity output
cosine_sim = tf.keras.layers.Dot(axes=1, normalize=True)([embed_1, embed_2])

# Define final model
similarity_model = tf.keras.Model(inputs=[input_1, input_2], outputs=cosine_sim)

similarity_model.compile(optimizer='adam', loss='mse')

import shutil
model_name = f"signal_matcher_resnet_triplet_loss"
similarity_model.save(f"{model_name}.keras")
# Should store threshold somewhere as well, it's our best similarity threshold when making predicitons
shutil.copy(f"{model_name}.keras", f"/lakehouse/default/Files/{model_name}.keras") 