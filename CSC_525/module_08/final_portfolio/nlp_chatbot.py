import nltk
import random
from training_data import training_data # Training data from a separate file.
from nltk.stem import WordNetLemmatizer 
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
# --------------------------------------------------------------------------------------
# Mental Health NLP Chatbot.
#
# This script implements a simple NLP-based chatbot for mental health support.
# It uses basic intent classification to respond to user input about anxiety, ADHD, and
# depression. The bot also detects crisis phrases and provides emergency resources if 
# needed.
#
# Usage:
#   $ python nlp_chatbot.py
# --------------------------------------------------------------------------------------
def preprocess(text, lemmatizer):
    """Tokenize and lemmatize input text for normalization.

    Args:
        text (str): The input text to preprocess.
        lemmatizer (WordNetLemmatizer): Lemmatizer instance.

    Returns:
        str: A single string of lemmatized tokens.
    """
    tokens = nltk.word_tokenize(text.lower()) # Convert to lowercase and tokenize.
    # Lemmatize each token and join into a single string.
    # Create a list of lemmatized tokens from the input text.
    lemmatized_tokens = []
    for token in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

def is_crisis(text, crisis_keywords):
    """Check if the input text contains any crisis keywords.

    Args:
        text (str): The user input text.
        crisis_keywords (list): List of crisis keywords.

    Returns:
        bool: True if a crisis phrase is found, otherwise False.
    """
    text = text.lower()
    # Check if any crisis keyword is present in the text.
    for phrase in crisis_keywords:
        if phrase in text:
            return True
    return False

def chatbot(lemmatizer, vectorizer, model, responses, crisis_keywords):
    """Main chatbot loop.

    Handles user input, checks for crisis situations, predicts intent, and prints 
    responses.
    """
    print(
        "Mental Health Bot: Hello, I'm here to support you. "
        "You can talk to me about anxiety, ADHD, or depression. "
        "Type 'bye' to exit."
    )
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("Mental Health Bot: Take care. You're not alone.")
            break
        if is_crisis(user_input, crisis_keywords):
            print("\nMental Health Bot: I'm really concerned about your safety.")
            print(
                "If you're thinking about hurting yourself or someone else, please call"
                " or text 988 — the Suicide & Crisis Lifeline. They are available 24/7.\n"
            )
            continue
        processed = preprocess(user_input, lemmatizer)
        user_vec = vectorizer.transform([processed])
        pred_intent = model.predict(user_vec)[0]
        # Simple keyword-based follow-up for smarter responses.
        if(pred_intent == "anxiety" and "sleep" in user_input.lower()):
            response = "Sleep issues are common with anxiety. Would you like some tips for better sleep?"
        elif(pred_intent == "depression" and "energy" in user_input.lower()):
            response = "Low energy can be a symptom of depression. Have you found anything that helps, even a little?"
        else:
            response = random.choice(responses.get(pred_intent, [
                "I'm not sure I understand. Could you tell me more?"
            ]))
        print("Mental Health Bot:", response)

def main():
    """Main driver function to set up and run the nlp chatbot."""
    # Download required NLTK data files for tokenization and lemmatization.
    nltk.download('punkt') # Required for nltk.word_tokenize, tokenizer models.
    nltk.download('wordnet') # Required for WordNetLemmatizer, WordNet database.
    lemmatizer = WordNetLemmatizer() # Initialize for word normalization.    
    # Preprocess training texts and extract their labels (intents).
    texts = [] # List of normalized training texts.
    for example in training_data:
        processed_text = preprocess(example["text"], lemmatizer)
        texts.append(processed_text)
    # Create a list of intent labels (targets) for each training example.
    labels = []
    for example in training_data:
        labels.append(example["intent"])
    # Convert training texts to TF-IDF feature vectors.
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    # Train a logistic regression model to classify user intent.
    model = LogisticRegression()
    model.fit(X, labels)
    # Map each intent to a list of possible bot responses.
    responses = {
        "greet": [
            "Hi there! I'm here to support you. How are you feeling today?",
            "Hello! How can I help you today?",
            "Hey! I'm here to listen. What's on your mind?",
            "Hi! How are you doing right now?"
        ],
        "anxiety": [
            "I'm sorry you're feeling anxious. Want to talk more about it?",
            "Anxiety can be tough. Would you like to share what's making you anxious?",
            "It's okay to feel anxious sometimes. Do you know what triggered it?",
            "I'm here for you. What helps you when you feel anxious?"
        ],
        "depression": [
            "That sounds really hard. Remember you're not alone in this.",
            "I'm sorry you're feeling down. Would you like to talk about it?",
            "Depression can feel overwhelming. Is there something specific on your mind?",
            "Thank you for sharing. What do you wish others understood about how you feel?"
        ],
        "adhd": [
            "Many people with ADHD feel that way. Would you like to talk about coping strategies?",
            "ADHD can make things challenging. What are you struggling with most right now?",
            "You're not alone. Are there any strategies that have helped you before?",
            "Would you like tips on focus or organization?"
        ],
        "goodbye": [
            "Take care of yourself. I'm here whenever you need support.",
            "Goodbye! Remember, you're not alone.",
            "See you next time. Wishing you well.",
            "Take care! Reach out whenever you need to talk."
        ]
    }
    # List of phrases that indicate a mental health crisis.
    crisis_keywords = [
        "kill myself", "want to die", "suicide", "end it all", "hurt myself",
        "hurt someone", "kill someone", "can't go on", "no reason to live"
    ]
    # Start the chatbot with all required resources.
    chatbot(lemmatizer, vectorizer, model, responses, crisis_keywords)

# The big red activation button.
if __name__ == "__main__":
    main() # Run the main driver function.
