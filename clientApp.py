# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from googletrans import Translator
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize Flask app
app = Flask(__name__)

# Function to translate Hindi text to English
def translate_hindi_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='hi', dest='en')
    return translated.text

# Function to load the BART model and tokenizer for text summarization
def load_summarization_model(model_name):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to summarize text using the BART model
def summarize_text(text, model, tokenizer, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Define the route for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # If the method is POST, process the uploaded file
        file = request.files['file']  # Get the file uploaded by the user
        hindi_text = file.read().decode('utf-8')  # Read and decode the file content

        # Translate the Hindi text to English
        english_text = translate_hindi_to_english(hindi_text)

        # Load the BART model for summarization
        summarization_model_name = "facebook/bart-large-cnn"
        summarization_model, summarization_tokenizer = load_summarization_model(summarization_model_name)

        # Summarize the translated English text
        summary_text = summarize_text(english_text, summarization_model, summarization_tokenizer)

        # Return the results as JSON
        return jsonify({"english": english_text, "summary": summary_text})

    # If the method is GET, show the index.html page
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app with debug mode on