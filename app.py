import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from datetime import datetime

app = Flask(__name__)

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Load and prepare data
data = pd.DataFrame({
    "id": [1, 2, 3],
    "title": ["Python", "Java", "C++"],
    "description": [
        "A high-level programming language known for its simplicity and readability",
        "A popular object-oriented programming language",
        "A powerful systems programming language"
    ]
})

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['title'])

def get_ai_explanation(query, results):
    try:
        # If no results found, ask AI for suggestions
        if not results:
            prompt = f"""The user searched for "{query}" and the available data is "{data}" but no exact matches were found in our programming language database. 
Now help the user find the most similar to this one with our availability. Provide the explanation in bullet points (not paragraphs). Limit the response to 200 words."""
        else:
            # Build context from the results
            context = "\n".join([f"- {r['title']}: {r['description']}" for r in results])
            prompt = f"""Query: "{query}"
            Search Results:
            {context}
            
            Please explain these programming languages in relation to the user's query. 
            Include key features and potential use cases that might be relevant.
            Format the explanation in bullet points (not paragraphs)."""
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful programming expert who explains technical concepts clearly and concisely."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"
def log_search(query, found_results):
    with open("search_log.csv", "a") as f:
        timestamp = datetime.now().isoformat()
        result_status = "found" if found_results else "not_found"
        f.write(f"{timestamp},{query},{result_status}\n")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.form.get("query", "").strip()
        if not query:
            return jsonify({"error": "Please enter a search query"}), 400

        # Perform TF-IDF search
        query_vector = vectorizer.transform([query])
        similarity_scores = cosine_similarity(tfidf_matrix, query_vector).flatten()
        data['similarity'] = similarity_scores
        
        # Get search results
        sorted_data = data.sort_values(by='similarity', ascending=False)
        results = sorted_data[sorted_data['similarity'] > 0].head(5).to_dict(orient="records")
        
        # Get AI explanation
        explanation = get_ai_explanation(query, results)
        
        # Log the search
        log_search(query, bool(results))
        
        return jsonify({
            "results": results,
            "explanation": explanation,
            "query": query
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)