# ==========================================================
# PHASE 5
# LIGHTWEIGHT RAG + TINYLLAMA
# FOR 8GB RAM LAPTOPS
# ==========================================================

import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

# ==========================================================
# DATASET
# ==========================================================

data = {

    "Poem Title": [
        "The Bangle Sellers",
        "Indian Weavers",
        "The Gift of India",
        "To My Children",
        "Humayun to Zobeida"
    ],

    "poem_text": [

        """
        Bangle sellers are we who bear
        Our shining loads to the temple fair.
        Who will buy these delicate bright
        Rainbow-tinted circles of light?
        """,

        """
        Weavers, weaving at break of day,
        Why do you weave a garment so gay?

        Blue as the wing of a halcyon wild,
        We weave the robes of a new-born child.

        Weavers, weaving at fall of night,
        Why do you weave a garment so bright?

        Like the plumes of a peacock, purple and green,
        We weave the marriage-veils of a queen.

        Weavers, weaving solemn and still,
        What do you weave in the moonlight chill?

        White as a feather and white as a cloud,
        We weave a dead man's funeral shroud.
        """,

        """
        Is there aught you need that my hands withhold,
        Rich gifts of raiment or grain or gold?
        """,

        """
        The love of a mother whose heart is pure,
        A blessing eternal and secure.
        """,

        """
        Love like a fragrant rose in bloom,
        Glows softly through the midnight gloom.
        """
    ]
}

df = pd.DataFrame(data)

# ==========================================================
# STEP 1
# LOAD EMBEDDING MODEL
# ==========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================================
# STEP 2
# CREATE EMBEDDINGS
# ==========================================================

print("Creating poem embeddings...")

poem_embeddings = embedding_model.encode(
    df["poem_text"].tolist()
)

print("Embeddings ready.")

# ==========================================================
# STEP 3
# USER QUERY
# ==========================================================

query = """
Write a poem similar to Indian Weavers.

Theme:
Cycle of life

Style:
Sarojini Naidu

Use question-answer format.
"""

# ==========================================================
# STEP 4
# EMBED QUERY
# ==========================================================

query_embedding = embedding_model.encode([query])

# ==========================================================
# STEP 5
# FIND SIMILAR POEMS
# ==========================================================

similarities = cosine_similarity(
    query_embedding,
    poem_embeddings
)

top_indices = similarities[0].argsort()[-2:][::-1]

print("\nRetrieved Poems:\n")

context = ""

for idx in top_indices:

    title = df.iloc[idx]["Poem Title"]

    score = similarities[0][idx]

    poem = df.iloc[idx]["poem_text"]

    print(f"{title} -> {score:.4f}")

    context += f"\n{poem}\n"

# ==========================================================
# STEP 6
# LOAD TINYLLAMA
# ==========================================================

print("\nLoading TinyLlama...")

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# ==========================================================
# STEP 7
# CHAT FORMAT PROMPT
# ==========================================================

prompt = f"""
<|system|>
You are Sarojini Naidu.

You write lyrical Indian poetry.

Use symbolism.
Use imagery.
Use musical rhythm.

<|user|>

Study these poems:

{context}

Write a NEW poem.

Requirements:

1. Similar to Indian Weavers
2. Question-answer style
3. Three stanzas
4. Theme = human life cycle
5. Do not copy

Only output poem.

<|assistant|>
"""

# ==========================================================
# STEP 8
# GENERATE
# ==========================================================

print("\nGenerating poem...\n")

result = generator(
    prompt,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7
)

# ==========================================================
# STEP 9
# OUTPUT
# ==========================================================

generated_text = result[0]["generated_text"]

print("\n")
print("=" * 60)
print("GENERATED POEM")
print("=" * 60)

print(generated_text)

print("\n")
print("=" * 60)
print("PHASE 5 TEST COMPLETE")
print("=" * 60)