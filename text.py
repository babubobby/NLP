import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextDenoiser:
    def __init__(self, remove_stopwords=True, remove_punctuation=True,
                 lowercase=True, lemmatize=True):
        """
        Initialize text denoising parameters
       
        Args:
            remove_stopwords (bool): Remove common stopwords
            remove_punctuation (bool): Remove punctuation marks
            lowercase (bool): Convert text to lowercase
            lemmatize (bool): Lemmatize words to their base form
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
       
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.lemmatize = lemmatize
       
        # Prepare stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
       
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
   
    def clean_text(self, text):
        """
        Denoise and clean the input text
       
        Args:
            text (str): Input text to be denoised
       
        Returns:
            str: Cleaned and denoised text
        """
        # Lowercase conversion
        if self.lowercase:
            text = text.lower()
       
        # Tokenization
        tokens = word_tokenize(text)
       
        # Remove punctuation
        if self.remove_punctuation:
            tokens = [token for token in tokens
                      if token not in string.punctuation]
       
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens
                      if token not in self.stop_words]
       
        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
       
        # Reconstruct text
        return ' '.join(tokens)
   
    def remove_special_characters(self, text):
        """
        Remove special characters and extra whitespaces
       
        Args:
            text (str): Input text
       
        Returns:
            str: Text with special characters removed
        """
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
       
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
       
        return text
   
    def remove_numbers(self, text):
        """
        Remove numerical characters
       
        Args:
            text (str): Input text
       
        Returns:
            str: Text with numbers removed
        """
        return re.sub(r'\d+', '', text)
   
    def denoise_text(self, text):
        """
        Comprehensive text denoising method
       
        Args:
            text (str): Input text to be denoised
       
        Returns:
            str: Fully denoised text
        """
        # Apply cleaning techniques in sequence
        text = self.remove_special_characters(text)
        text = self.remove_numbers(text)
        text = self.clean_text(text)
       
        return text

# Example usage
def main():
    # Initialize denoiser
    denoiser = TextDenoiser(
        remove_stopwords=True,
        remove_punctuation=True,
        lowercase=True,
        lemmatize=True
    )
   
    # Sample noisy text
    noisy_text = "Hello123! This is a sample noisy text, with some @special characters and numbers."
   
    # Denoise the text
    cleaned_text = denoiser.denoise_text(noisy_text)
    print("Original Text:", noisy_text)
    print("Denoised Text:", cleaned_text)

if __name__ == "__main__":
    main()