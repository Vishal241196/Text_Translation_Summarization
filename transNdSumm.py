# Import necessary libraries
from googletrans import Translator  # For translating text from Hindi to English
import torch  # PyTorch, used by the transformers library
from transformers import BartForConditionalGeneration, BartTokenizer  # For text summarization

# Function to translate text from Hindi to English
def translate_hindi_to_english(text):
    translator = Translator()  # Initialize the Translator
    translated = translator.translate(text, src='hi', dest='en')  # Translate the text
    return translated.text  # Return the translated text

# Function to load a pre-trained BART model and its tokenizer for summarization
def load_summarization_model(model_name):
    model = BartForConditionalGeneration.from_pretrained(model_name)  # Load the model
    tokenizer = BartTokenizer.from_pretrained(model_name)  # Load the tokenizer
    return model, tokenizer  # Return both the model and tokenizer

# Function to summarize text using the BART model
def summarize_text(text, model, tokenizer, max_length=150):
    # Encode the text for the model
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    # Generate summary
    outputs = model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode the generated tokens to text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary  # Return the summarized text

if __name__ == '__main__':
    # Path to the file containing the Hindi article
    inPath = r"C:\Users\admin\TranslateAndSummarizeText\HindiArticle.txt"
    
    # Open and read the Hindi article file
    with open(inPath, 'r', encoding='utf-8') as file:
        hindi_text = file.read()  # Read the entire contents of the file

    # Translate the Hindi text to English
    english_text = translate_hindi_to_english(hindi_text)

    # Name of the pre-trained BART model for summarization
    summarization_model_name = "facebook/bart-large-cnn"
    # Load the summarization model and its tokenizer
    summarization_model, summarization_tokenizer = load_summarization_model(summarization_model_name)

    # Summarize the translated English text
    summary_text = summarize_text(english_text, summarization_model, summarization_tokenizer)

    # Print the original Hindi text, its English translation, and the summary
    print("Hindi: ", hindi_text)
    # Calculate and print the total number of words in the Hindi text
    words_hin = ' '.join(hindi_text).split()
    total_words_hin = len(words_hin)
    print("Total words in Hindi text", total_words_hin)

    print("English: ", english_text)
    # Calculate and print the total number of words in the English translation
    words_eng = ' '.join(english_text).split()
    total_words_eng = len(words_eng)
    print("Total words in English text", total_words_eng)

    print("Summary: ", summary_text)
    # Calculate and print the total number of words in the summary
    words_sum = ' '.join(summary_text).split()
    total_words_sum = len(words_sum)
    print("Total words in summary text", total_words_sum)
    


    
    
