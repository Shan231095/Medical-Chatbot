from flask import Flask, render_template, request
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="GEMINI_API_KEY")

# Global variables
chunks = []
vectorizer = None

# Load PDF and prepare chunks
def load_pdf_chunks(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = [doc.page_content for doc in docs]
    return chunks

pdf_path = r"E:\Shanmathi\Hope AI course Tamil\RAG\Medical Chatbot - flask\Heart Disease-Full Text.pdf"
chunks = load_pdf_chunks(pdf_path)
vectorizer = TfidfVectorizer().fit_transform(chunks)

# Retrieve relevant context
def retrieve_context(query, top_k=3):
    query_vec = TfidfVectorizer().fit(chunks).transform([query])
    similarity = cosine_similarity(query_vec, vectorizer)
    top_indices = similarity.argsort()[0][-top_k:][::-1]
    retrieved = "\n\n".join([chunks[i] for i in top_indices])
    return retrieved

# Ask Gemini
def ask_gemini(context, user_query):
    prompt = (
        f"You are a medical assistant specialized in heart diseases.\n"
        f"Use the following context to answer the user's question:\n\n"
        f"{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    user_message = ""
    bot_reply = ""

    if request.method == "POST":
        user_message = request.form.get("user_input")

        if user_message:
            context = retrieve_context(user_message)
            bot_reply = ask_gemini(context, user_message)

    return render_template("index.html", user_message=user_message, bot_reply=bot_reply)

if __name__ == "__main__":
    app.run(debug=True)
