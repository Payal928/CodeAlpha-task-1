
import streamlit as st
import json
import re
import nltk

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Chatbot")
st.write("Ask me anything about online shopping!")

st.divider()


# ---------------- NLP PREPROCESSING ----------------

# NLTK tokenizer and stemmer
tokenizer = RegexpTokenizer(r"\w+")
stemmer = PorterStemmer()

# Common English stopwords
stop_words = {
    "a", "an", "the", "is", "am", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "from", "with", "by", "and", "or", "but",
    "do", "does", "did", "have", "has", "had",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "our", "their", "this", "that",
    "what", "how", "when", "where", "who", "which",
    "can", "could", "would", "should", "will"
}


def preprocess_text(text):
    """
    Clean text, tokenize, remove stopwords
    and apply stemming.
    """

    text = text.lower()

    # Tokenization
    tokens = tokenizer.tokenize(text)

    # Stopword removal and stemming
    processed_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(processed_tokens)


# ---------------- LOAD FAQ DATASET ----------------

@st.cache_data
def load_faqs():

    with open("faqs.json", "r", encoding="utf-8") as file:
        faqs = json.load(file)

    return faqs


faqs = load_faqs()


# ---------------- TF-IDF MODEL ----------------

@st.cache_resource
def create_model():

    # Extract FAQ questions
    questions = [
        faq["question"]
        for faq in faqs
    ]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        lowercase=False,
        ngram_range=(1, 2)
    )

    # Convert FAQ questions into vectors
    faq_vectors = vectorizer.fit_transform(questions)

    return vectorizer, faq_vectors


vectorizer, faq_vectors = create_model()


# ---------------- FAQ MATCHING FUNCTION ----------------

def get_answer(user_question):

    # Convert user question into TF-IDF vector
    user_vector = vectorizer.transform([user_question])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Find the index of the most similar FAQ
    best_match_index = similarity_scores.argmax()

    # Get the similarity score
    best_score = similarity_scores[0][best_match_index]

    # Minimum similarity threshold
    threshold = 0.15

    if best_score < threshold:

        return {
            "answer": (
                "Sorry, I couldn't find a relevant answer "
                "to your question. Please try asking differently "
                "or contact customer support."
            ),
            "matched_question": None,
            "score": best_score
        }

    # Get the matching FAQ
    matched_faq = faqs[best_match_index]

    return {
        "answer": matched_faq["answer"],
        "matched_question": matched_faq["question"],
        "score": best_score
    }


# ---------------- CHAT INTERFACE ----------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversations
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("matched"):
            st.caption(
                f"Matched FAQ: {message['matched']}"
            )

# User input
user_input = st.chat_input(
    "Type your question here..."
)

if user_input:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate chatbot response
    with st.chat_message("assistant"):

        with st.spinner("Finding the best answer..."):

            result = get_answer(user_input)

        st.markdown(result["answer"])

        if result["matched_question"]:

            st.caption(
                f"Matched FAQ: {result['matched_question']}"
            )

            st.caption(
                f"Similarity Score: {result['score']:.2%}"
            )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "matched": result["matched_question"]
    })


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("ℹ️ About Chatbot")

    st.write(
        "This chatbot uses NLP preprocessing, "
        "TF-IDF vectorization and cosine similarity "
        "to find answers from a predefined FAQ dataset."
    )

    st.divider()

    st.subheader("Available FAQs")

    st.write(f"Total FAQs: {len(faqs)}")

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Built using Python, NLTK and Streamlit.")
