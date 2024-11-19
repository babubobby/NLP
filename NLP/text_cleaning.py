import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming and Lemmatization
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(ps.stem(word)) for word in words]
    return ' '.join(words)

# Example usage with the paragraph
sample_text = "   Natural   language processing  (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages. As such, NLP is related to the area of human-computer interaction.  Many  challenges  in  NLP  involve  natural  language  understanding, that is, enabling computers to derive meaning from human or natural language input, and others involve natural language generation."
cleaned_text = clean_text(sample_text)
print(cleaned_text)
