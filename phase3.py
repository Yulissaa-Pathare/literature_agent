id="r2db9c"
# ==========================================
# PHASE 3 — EMBEDDING BASED SEMANTIC ENGINE
# ==========================================

# ---------- IMPORT LIBRARIES ----------

import pandas as pd
import numpy as np

# SentenceTransformer generates semantic embeddings
from sentence_transformers import SentenceTransformer

# cosine_similarity compares meaning similarity
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("sarojini_naidu_poetry_dataset.csv")

# ==========================================
# MANUAL POEM TEXTS
# ==========================================

# We manually attach poem excerpts because
# embeddings require actual language text

poem_texts = {

    "The Bangle Sellers": """
    Bangle sellers are we who bear
    Our shining loads to the temple fair.
    Who will buy these delicate, bright
    Rainbow-tinted circles of light?
    """,

    "Indian Weavers": """
    Weavers, weaving at break of day,
    Why do you weave a garment so gay?
    Blue as the wing of a halcyon wild,
    We weave the robes of a new-born child.
    """,

    "The Gift of India": """
    Is there aught you need that my hands withhold,
    Rich gifts of raiment or grain or gold?
    Lo! I have flung to the East and West
    Priceless treasures torn from my breast.
    """,

    "To My Children": """
    The love of a mother whose heart is pure,
    A blessing eternal and secure.
    May your lives shine bright and free,
    Like stars above the silent sea.
    """,

    "Humayun to Zobeida": """
    Love, like a fragrant rose in bloom,
    Glows softly through the midnight gloom.
    O beloved, your eyes shine bright,
    Like moonlit stars in silent night.
    """
}

# Add poem text to dataframe
df['poem_text'] = df['Poem Title'].map(poem_texts)

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

# all-MiniLM-L6-v2 is a lightweight
# semantic embedding model

# It converts sentences into vectors
# containing semantic meaning

model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# CREATE EMBEDDINGS
# ==========================================

# Convert every poem into semantic vectors

poem_embeddings = model.encode(

    df['poem_text'].tolist()

)

# ==========================================
# SHOW EMBEDDING INFORMATION
# ==========================================

print("\n========== EMBEDDING INFORMATION ==========\n")

print("Number of Poems:", len(poem_embeddings))

print("Embedding Vector Shape:", poem_embeddings[0].shape)

# ==========================================
# TEST NEW POEM
# ==========================================

new_poem = """

Silver moonlight softly glows,
Through the garden where jasmine grows.
Silent rivers gently stream,
Like fragments of a timeless dream.

"""

# ==========================================
# GENERATE EMBEDDING FOR NEW POEM
# ==========================================

new_embedding = model.encode([new_poem])

# ==========================================
# CALCULATE SIMILARITY
# ==========================================

# cosine similarity measures semantic closeness

similarities = cosine_similarity(

    new_embedding,
    poem_embeddings

)

# ==========================================
# FIND MOST SIMILAR POEM
# ==========================================

most_similar_index = np.argmax(similarities)

most_similar_poem = df.iloc[most_similar_index]['Poem Title']

similarity_score = similarities[0][most_similar_index]

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n========== SEMANTIC SIMILARITY RESULTS ==========\n")

print("Input Poem:\n")

print(new_poem)

print("\nMost Similar Sarojini Naidu Poem:\n")

print(most_similar_poem)

print("\nSimilarity Score:\n")

print(round(similarity_score, 4))

# ==========================================
# SHOW ALL SIMILARITY SCORES
# ==========================================

print("\n========== ALL POEM SIMILARITIES ==========\n")

for i in range(len(df)):

    title = df.iloc[i]['Poem Title']

    score = similarities[0][i]

    print(f"{title} --> Similarity Score: {score:.4f}")

