import pdfplumber  # type: ignore
import torch  # type: ignore
from transformers import T5Tokenizer, T5ForConditionalGeneration  # type: ignore
import nltk  # type: ignore
nltk.download('punkt_tab')
from nltk.corpus import stopwords  # type: ignore
nltk.download('stopwords')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure there is text on the page
                all_text += page_text
    return all_text

# Function to create the T5 model and tokenizer
def create_t5_model():
    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    return tokenizer, model

# Function to summarize the context using the T5 model
def summarize_text(tokenizer, model, context):
    # Prepare the text for summarization
    input_text = f"summarize: {context}"
    
    # Tokenize the input
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to suggest questions based on the summarized context
def suggest_questions_from_summary(summary):
    # Tokenize the text and remove stopwords
    words = nltk.word_tokenize(summary.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Generate a variety of question templates for each of the 5 common keywords
    question_templates = [
        f"What is {words[0]}?",
        f"Why is {words[1]} important?",
        f"How does {words[2]} impact the overall theme?",
        f"Can you explain the role of {words[3]} in this context?",
        f"What are the key points about {words[4]}?"
    ]
    
    return question_templates

# Function to answer a query based on the PDF context
def answer_query(tokenizer, model, context, question):
    # Prepare input by formatting it into a question-answering prompt
    input_text = f"question: {question} context: {context}"
    
    # Tokenize the input
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate the answer
    outputs = model.generate(inputs, max_length=600, num_beams=4, early_stopping=True)
    
    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check if the answer is empty or irrelevant and return "No answer found" if necessary
    if not answer or len(answer.split()) < 3:
        return "No answer found"
    
    return answer

# Main function
def main():
    # Path to the PDF file
    pdf_path = r"/Users/macbook/Desktop/IPQS/backend/anml.pdf"
    
    # Extract text from the PDF
    context = extract_text_from_pdf(pdf_path)
    
    # Initialize T5 model and tokenizer
    tokenizer, model = create_t5_model()
    
    # Summarize the text to get key points
    summary = summarize_text(tokenizer, model, context)
    print("\nSummary of the document:")
    print(summary)
    
    # Suggest questions based on the summary
    suggested_questions = suggest_questions_from_summary(summary)
    print("\nHere are 5 suggested questions based on the summary:")
    for i, question in enumerate(suggested_questions, 1):
        print(f"{i}. {question}")
    
    # Loop for continuous querying until the user types 'exit'
    while True:
        # Take user query as input
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        
        # Exit if the user types 'exit'
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        
        # Answer the query based on the extracted context
        answer = answer_query(tokenizer, model, context, query)
        
        # Print the answer
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

