# ==========================================
# SAROJINI NAIDU FEATURE ENGINEERING PIPELINE
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

from textblob import TextBlob

import numpy as np

# ==========================================
# DOWNLOAD NLP DATA
# ==========================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("sarojini_naidu_poetry_dataset.csv")

# ==========================================
# MANUAL POEM TEXTS
# ==========================================

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
# TEXT CLEANING
# ==========================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    words = word_tokenize(text)

    cleaned_words = []

    for word in words:

        if word.isalpha() and word not in stop_words:
            cleaned_words.append(word)

    return " ".join(cleaned_words)

# Apply cleaning
df['cleaned_text'] = df['poem_text'].apply(clean_text)

# ==========================================
# FEATURE ENGINEERING FUNCTIONS
# ==========================================

# ---------- WORD COUNT ----------

def word_count(text):

    words = word_tokenize(text)

    return len(words)

# ---------- VOCABULARY RICHNESS ----------

def vocabulary_richness(text):

    words = word_tokenize(text)

    unique_words = set(words)

    if len(words) == 0:
        return 0

    return len(unique_words) / len(words)

# ---------- AVERAGE LINE LENGTH ----------

def average_line_length(text):

    lines = text.split('\n')

    line_lengths = []

    for line in lines:

        words = word_tokenize(line)

        if len(words) > 0:
            line_lengths.append(len(words))

    if len(line_lengths) == 0:
        return 0

    return sum(line_lengths) / len(line_lengths)

# ---------- REPETITION SCORE ----------

def repetition_score(text):

    words = word_tokenize(text)

    if len(words) == 0:
        return 0

    unique_words = set(words)

    repeated_words = len(words) - len(unique_words)

    return repeated_words / len(words)

# ---------- SENTIMENT SCORE ----------

def sentiment_score(text):

    blob = TextBlob(text)

    return blob.sentiment.polarity

# ==========================================
# APPLY FEATURE EXTRACTION
# ==========================================

df['word_count'] = df['poem_text'].apply(word_count)

df['vocab_richness'] = df['poem_text'].apply(vocabulary_richness)

df['avg_line_length'] = df['poem_text'].apply(average_line_length)

df['repetition_score'] = df['poem_text'].apply(repetition_score)

df['sentiment_score'] = df['poem_text'].apply(sentiment_score)

# ==========================================
# SHOW EXTRACTED FEATURES
# ==========================================

print("\n========== EXTRACTED FEATURES ==========\n")

print(df[[

    'Poem Title',
    'word_count',
    'vocab_richness',
    'avg_line_length',
    'repetition_score',
    'sentiment_score'

]])

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer()

X_text = vectorizer.fit_transform(df['cleaned_text']).toarray()

# ==========================================
# COMBINE TF-IDF + HANDCRAFTED FEATURES
# ==========================================

manual_features = df[[

    'word_count',
    'vocab_richness',
    'avg_line_length',
    'repetition_score',
    'sentiment_score'

]].values

# Combine all features
X = np.hstack((X_text, manual_features))

# Labels
y = df['Tone']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)

# ==========================================
# TEST MODEL
# ==========================================

predictions = model.predict(X_test)

print("\n========== MODEL REPORT ==========\n")

print(classification_report(y_test, predictions))

# ==========================================
# TEST NEW POEM
# ==========================================

new_poem = """

Golden flowers softly sway
Beside the silent river grey,
Moonlit winds begin to rise
Under calm and silver skies.

"""

# Clean poem
cleaned_poem = clean_text(new_poem)

# TF-IDF vector
poem_vector_text = vectorizer.transform([cleaned_poem]).toarray()

# Extract manual features
new_features = np.array([

    word_count(new_poem),
    vocabulary_richness(new_poem),
    average_line_length(new_poem),
    repetition_score(new_poem),
    sentiment_score(new_poem)

]).reshape(1, -1)

# Combine vectors
final_vector = np.hstack((poem_vector_text, new_features))

# Predict
prediction = model.predict(final_vector)

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n========== NEW POEM ANALYSIS ==========\n")

print("Predicted Tone:", prediction[0])

print("\n========== FEATURE VALUES ==========\n")

print("Word Count:", word_count(new_poem))

print("Vocabulary Richness:", vocabulary_richness(new_poem))

print("Average Line Length:", average_line_length(new_poem))

print("Repetition Score:", repetition_score(new_poem))

print("Sentiment Score:", sentiment_score(new_poem))

