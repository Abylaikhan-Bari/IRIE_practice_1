import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")

# Sample English text
text_en = "Natural Language Processing (NLP) is an exciting field of artificial intelligence."

# --- Task 1: Tokenization ---
# NLTK Tokenization
tokens_nltk_en = word_tokenize(text_en)

# spaCy Tokenization
doc_en = nlp_en(text_en)
tokens_spacy_en = [token.text for token in doc_en]

# --- Task 2: Stop-word Removal ---
# NLTK Stop-word removal
stop_words_en = set(stopwords.words('english'))
filtered_words_nltk_en = [word for word in tokens_nltk_en if word.lower() not in stop_words_en]

# spaCy Stop-word removal
filtered_words_spacy_en = [token.text for token in doc_en if not token.is_stop]

# --- Task 3: Comparison of Results ---
print("\n🔹 English Tokenization Results:")
print("NLTK:", tokens_nltk_en)
print("spaCy:", tokens_spacy_en)

print("\n🔹 English Stop-word Removal:")
print("NLTK:", filtered_words_nltk_en)
print("spaCy:", filtered_words_spacy_en)

# --- Task 4: Russian Text Processing ---
text_ru = "Обработка естественного языка — это захватывающая область искусственного интеллекта."

# NLTK Tokenization for Russian
tokens_nltk_ru = word_tokenize(text_ru, language="russian")

# spaCy Tokenization for Russian
doc_ru = nlp_ru(text_ru)
tokens_spacy_ru = [token.text for token in doc_ru]

# NLTK Stop-word removal for Russian
stop_words_ru = set(stopwords.words('russian'))
filtered_words_nltk_ru = [word for word in tokens_nltk_ru if word.lower() not in stop_words_ru]

# spaCy Stop-word removal for Russian
filtered_words_spacy_ru = [token.text for token in doc_ru if not token.is_stop]

# --- Task 5: Display Russian Results ---
print("\n🔹 Russian Tokenization Results:")
print("NLTK:", tokens_nltk_ru)
print("spaCy:", tokens_spacy_ru)

print("\n🔹 Russian Stop-word Removal:")
print("NLTK:", filtered_words_nltk_ru)
print("spaCy:", filtered_words_spacy_ru)
