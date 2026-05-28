
# ==========================================
# SAROJINI NAIDU LITERARY ANALYZER
# ==========================================

# ---------- IMPORT LIBRARIES ----------

import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==========================================
# DOWNLOAD REQUIRED NLP DATA
# ==========================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================================
# STEP 1 — LOAD YOUR CSV DATASET
# ==========================================

# Make sure this CSV file is in the SAME folder
# as this Python file

df = pd.read_csv("sarojini_naidu_poetry_dataset.csv")

# ==========================================
# STEP 2 — SHOW COLUMN NAMES
# ==========================================

print("\n========== DATASET COLUMNS ==========\n")
print(df.columns)

# ==========================================
# STEP 3 — CREATE MANUAL POEM TEXT DATA
# ==========================================

# Since your CSV mainly contains literary metadata,
# we manually add poem excerpts here for NLP learning

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

# ==========================================
# STEP 4 — ADD POEM TEXT COLUMN
# ==========================================

# Match poem titles from CSV to manual poem text

df['poem_text'] = df['Poem Title'].map(poem_texts)

# ==========================================
# STEP 5 — CHECK DATASET
# ==========================================

print("\n========== DATASET PREVIEW ==========\n")

print(df[['Poem Title', 'Tone', 'poem_text']])

# ==========================================
# STEP 6 — TEXT CLEANING
# ==========================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Safety check
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    cleaned_words = []

    for word in words:

        if word.isalpha() and word not in stop_words:
            cleaned_words.append(word)

    return " ".join(cleaned_words)

# Apply cleaning
df['cleaned_text'] = df['poem_text'].apply(clean_text)

# ==========================================
# STEP 7 — TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['cleaned_text'])

# Labels to predict
y = df['Tone']

# ==========================================
# STEP 8 — TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# ==========================================
# STEP 9 — TRAIN MODEL
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

# ==========================================
# STEP 10 — MODEL PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\n========== MODEL REPORT ==========\n")

print(classification_report(y_test, predictions))

# ==========================================
# STEP 11 — TEST NEW POEM
# ==========================================

new_poem = """

Golden lamps are softly gleaming
Across the silent river streaming,
Moonlit flowers gently sway
In the calm of evening grey.

"""

# Clean poem
cleaned_poem = clean_text(new_poem)

# Convert to TF-IDF vector
poem_vector = vectorizer.transform([cleaned_poem])

# Predict tone
prediction = model.predict(poem_vector)

# ==========================================
# STEP 12 — FINAL OUTPUT
# ==========================================

print("\n========== NEW POEM ANALYSIS ==========\n")

print("Predicted Tone:", prediction[0])

print("\n========== CLEANED TEST POEM ==========\n")

print(cleaned_poem)
