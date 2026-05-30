# ==========================================================
# PHASE 5
# RAG + LLM GENERATION
# SAROJINI NAIDU STYLE TEST
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
# EMBEDDINGS
# ==========================================================

print("\nCreating embeddings...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

poem_embeddings = embedding_model.encode(
    df["poem_text"].tolist()
)

print("Embeddings ready.")

# ==========================================================
# USER QUERY
# ==========================================================

query = """
Write a poem similar to Indian Weavers.

Theme:
cycle of human life

Style:
Sarojini Naidu

Use:
question-answer structure
imagery
symbolism
musical rhythm

Do not copy.
"""

# ==========================================================
# QUERY EMBEDDING
# ==========================================================

query_embedding = embedding_model.encode([query])

# ==========================================================
# RETRIEVAL
# ==========================================================

similarities = cosine_similarity(
    query_embedding,
    poem_embeddings
)

top_k = 2

top_indices = similarities[0].argsort()[-top_k:][::-1]

print("\nRetrieved Poems:\n")

context = ""

for idx in top_indices:

    title = df.iloc[idx]["Poem Title"]

    score = similarities[0][idx]

    poem = df.iloc[idx]["poem_text"]

    print(f"{title}  |  score = {score:.4f}")

    context += f"\n\nTITLE: {title}\n{poem}\n"

# ==========================================================
# LOAD BETTER OPEN MODEL
# ==========================================================

print("\nLoading model...\n")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto"
)

# ==========================================================
# PROMPT
# ==========================================================

prompt = f"""
You are an expert literary scholar.

Study the poems below carefully.

{context}

Task:

Write a NEW poem in the style of Sarojini Naidu.

Requirements:

1. Similar to Indian Weavers
2. Question-answer format
3. Three stanzas
4. Human life symbolism
5. Musical rhythm
6. Rich imagery
7. Do not copy lines

Output ONLY the poem.
"""

# ==========================================================
# GENERATE
# ==========================================================

result = generator(
    prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.95
)

# ==========================================================
# OUTPUT
# ==========================================================

print("\n")
print("="*60)
print("GENERATED POEM")
print("="*60)

print(result[0]["generated_text"])