from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber  # type: ignore
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk  # type: ignore

# Replace 'YOUR_TOKEN' with your actual GitHub personal access token
headers = {
    'Authorization': 'token YOUR_TOKEN',
    'Accept': 'application/vnd.github.v3+json'
}


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load T5 model and tokenizer once at startup
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text
    return all_text

# Function to summarize the context using the T5 model
def summarize_text(context):
    input_text = f"summarize: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to suggest questions based on the summarized context
def suggest_questions_from_summary(summary):
    words = nltk.word_tokenize(summary.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    question_templates = []
    if len(words) >= 5:  # Ensure we have enough words to create questions
        question_templates = [
            f"What is {words[0]}?",
            f"Why is {words[1]} important?",
            f"How does {words[2]} impact the overall theme?",
            f"Can you explain the role of {words[3]} in this context?",
            f"What are the key points about {words[4]}?"
        ]
    return question_templates

# Function to answer a query based on the PDF context
def answer_query(context, question):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=600, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if not answer or len(answer.split()) < 3:
        return "No answer found"
    
    return answer

# Route to handle PDF upload and query
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    pdf_file = request.files['pdf']  # Get uploaded PDF
    question = request.form.get('question')  # Get question from form

    try:
        context = extract_text_from_pdf(pdf_file)
        summary = summarize_text(context)
        suggested_questions = suggest_questions_from_summary(summary)

        answer = answer_query(context, question) if question else None

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'summary': summary,
        'suggested_questions': suggested_questions,
        'answer': answer
    })

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)







