id="ztl96r"
# ==========================================
# PHASE 4 — SEMANTIC RETRIEVAL ENGINE
# ==========================================

# ---------- IMPORT LIBRARIES ----------

import pandas as pd
import numpy as np

# SentenceTransformer creates semantic embeddings
from sentence_transformers import SentenceTransformer

# cosine_similarity compares semantic meaning
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("sarojini_naidu_poetry_dataset.csv")

# ==========================================
# ADD POEM TEXTS
# ==========================================

# In future phases,
# this will come from full datasets or databases

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

# Add poem text column
df['poem_text'] = df['Poem Title'].map(poem_texts)

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

# This transformer model converts text
# into semantic vector representations

model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# GENERATE POEM EMBEDDINGS
# ==========================================

# Convert all poems into embeddings

poem_embeddings = model.encode(

    df['poem_text'].tolist()

)

# ==========================================
# USER QUERY
# ==========================================

# Simulated user literary request

query = """

A poem about love, moonlight,
beauty, longing, and romance

"""

# ==========================================
# CONVERT QUERY INTO EMBEDDING
# ==========================================

query_embedding = model.encode([query])

# ==========================================
# CALCULATE SEMANTIC SIMILARITY
# ==========================================

# Compare query meaning
# against all poems

similarities = cosine_similarity(

    query_embedding,
    poem_embeddings

)

# ==========================================
# RETRIEVE TOP MATCHES
# ==========================================

# Number of poems to retrieve

top_k = 3

# Sort similarities in descending order

top_indices = similarities[0].argsort()[-top_k:][::-1]

# ==========================================
# DISPLAY RETRIEVAL RESULTS
# ==========================================

print("\n========== USER QUERY ==========\n")

print(query)

print("\n========== TOP RETRIEVED POEMS ==========\n")

for rank, index in enumerate(top_indices):

    title = df.iloc[index]['Poem Title']

    tone = df.iloc[index]['Tone']

    score = similarities[0][index]

    poem_text = df.iloc[index]['poem_text']

    print(f"\nRANK {rank + 1}")

    print("----------------------------")

    print("Poem Title:", title)

    print("Tone:", tone)

    print("Similarity Score:", round(score, 4))

    print("\nPoem Excerpt:\n")

    print(poem_text)

# ==========================================
# EXPLAIN WHAT HAPPENED
# ==========================================

print("\n========== RETRIEVAL COMPLETE ==========\n")

print("The system semantically searched")

print("the literary embedding space")

print("and retrieved the most")

print("meaningfully relevant poems.")

