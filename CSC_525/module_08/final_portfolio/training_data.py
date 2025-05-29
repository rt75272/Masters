# --------------------------------------------------------------------------------------
# Training data for NLP chatbot.
# 
# Usage:
#   from training_data import training_data, responses, crisis_keywords
#-------------------------------------------------------------------------------------- 
training_data = [
    # GREETINGS
    {"intent": "greet", "text": "hello"},
    {"intent": "greet", "text": "hi"},
    {"intent": "greet", "text": "hey there"},
    {"intent": "greet", "text": "good morning"},
    {"intent": "greet", "text": "good evening"},
    {"intent": "greet", "text": "howdy"},
    {"intent": "greet", "text": "hey"},
    {"intent": "greet", "text": "what's up"},
    {"intent": "greet", "text": "greetings"},
    {"intent": "greet", "text": "yo"},
    {"intent": "greet", "text": "hi there"},
    {"intent": "greet", "text": "hey bot"},
    {"intent": "greet", "text": "hello there"},
    {"intent": "greet", "text": "hiya"},
    {"intent": "greet", "text": "sup"},
    # ANXIETY
    {"intent": "anxiety", "text": "I feel anxious and can't sleep"},
    {"intent": "anxiety", "text": "I'm having a panic attack"},
    {"intent": "anxiety", "text": "My thoughts are racing"},
    {"intent": "anxiety", "text": "I can't stop worrying"},
    {"intent": "anxiety", "text": "I feel nervous all the time"},
    {"intent": "anxiety", "text": "My heart is pounding and I can't relax"},
    {"intent": "anxiety", "text": "I feel overwhelmed"},
    {"intent": "anxiety", "text": "I can't breathe when I'm anxious"},
    {"intent": "anxiety", "text": "I get sweaty and shaky"},
    {"intent": "anxiety", "text": "I worry about everything"},
    {"intent": "anxiety", "text": "I have trouble calming down"},
    {"intent": "anxiety", "text": "I feel tense"},
    {"intent": "anxiety", "text": "I can't concentrate because of anxiety"},
    {"intent": "anxiety", "text": "I feel like something bad will happen"},
    {"intent": "anxiety", "text": "I get anxious in social situations"},
    {"intent": "anxiety", "text": "I have anxiety attacks at night"},
    {"intent": "anxiety", "text": "I can't control my anxiety"},
    {"intent": "anxiety", "text": "I feel restless and uneasy"},
    # DEPRESSION
    {"intent": "depression", "text": "I feel so down all the time"},
    {"intent": "depression", "text": "Nothing feels good anymore"},
    {"intent": "depression", "text": "I don't want to do anything"},
    {"intent": "depression", "text": "I feel hopeless"},
    {"intent": "depression", "text": "I have no energy"},
    {"intent": "depression", "text": "I can't get out of bed"},
    {"intent": "depression", "text": "I feel empty inside"},
    {"intent": "depression", "text": "I don't enjoy things I used to"},
    {"intent": "depression", "text": "I feel worthless"},
    {"intent": "depression", "text": "I cry for no reason"},
    {"intent": "depression", "text": "I feel like a burden"},
    {"intent": "depression", "text": "I can't find motivation"},
    {"intent": "depression", "text": "I feel numb"},
    {"intent": "depression", "text": "I feel like giving up"},
    {"intent": "depression", "text": "I feel alone even around people"},
    {"intent": "depression", "text": "I have trouble eating because I'm sad"},
    {"intent": "depression", "text": "I can't sleep because I'm sad"},
    # ADHD
    {"intent": "adhd", "text": "I can't focus at all"},
    {"intent": "adhd", "text": "I'm so easily distracted"},
    {"intent": "adhd", "text": "I have trouble sitting still"},
    {"intent": "adhd", "text": "I keep forgetting things"},
    {"intent": "adhd", "text": "I can't finish tasks"},
    {"intent": "adhd", "text": "My mind is always wandering"},
    {"intent": "adhd", "text": "I lose things all the time"},
    {"intent": "adhd", "text": "I get bored quickly"},
    {"intent": "adhd", "text": "I interrupt people a lot"},
    {"intent": "adhd", "text": "I can't organize my work"},
    {"intent": "adhd", "text": "I procrastinate a lot"},
    {"intent": "adhd", "text": "I forget appointments"},
    {"intent": "adhd", "text": "I fidget constantly"},
    {"intent": "adhd", "text": "I can't pay attention in meetings"},
    {"intent": "adhd", "text": "I feel restless at work"},
    # GOODBYE
    {"intent": "goodbye", "text": "bye"},
    {"intent": "goodbye", "text": "see you later"},
    {"intent": "goodbye", "text": "thanks, bye"},
    {"intent": "goodbye", "text": "goodbye"},
    {"intent": "goodbye", "text": "talk to you later"},
    {"intent": "goodbye", "text": "see ya"},
    {"intent": "goodbye", "text": "catch you later"},
    {"intent": "goodbye", "text": "farewell"},
    {"intent": "goodbye", "text": "I'm leaving now"},
    {"intent": "goodbye", "text": "I have to go"},
    {"intent": "goodbye", "text": "bye for now"},
    {"intent": "goodbye", "text": "see you soon"},
    {"intent": "goodbye", "text": "talk soon"},
]

# Bot responses for different intents.
responses = {
    "greet": [
        "Hi there! I'm here to support you. How are you feeling today?",
        "Hello! How can I help you today?",
        "Hey! I'm here to listen. What's on your mind?",
        "Hi! How are you doing right now?",
        "It's good to hear from you. How can I support you today?",
        "Welcome! Feel free to share what's on your mind.",
        "Hey there! How are you feeling?",
        "Hi! Is there something you'd like to talk about?"
    ],
    "anxiety": [
        "I'm sorry you're feeling anxious. Want to talk more about it?",
        "Anxiety can be tough. Would you like to share what's making you anxious?",
        "It's okay to feel anxious sometimes. Do you know what triggered it?",
        "I'm here for you. What helps you when you feel anxious?",
        "Would you like some grounding techniques for anxiety?",
        "Breathing exercises can sometimes help. Want to try one together?",
        "You're not alone in feeling this way. Many people experience anxiety.",
        "Is there something specific that's making you anxious right now?",
        "Would you like to talk about what helps you calm down?"
    ],
    "depression": [
        "That sounds really hard. Remember you're not alone in this.",
        "I'm sorry you're feeling down. Would you like to talk about it?",
        "Depression can feel overwhelming. Is there something specific on your mind?",
        "Thank you for sharing. What do you wish others understood about how you feel?",
        "Sometimes talking about it can help. I'm here to listen.",
        "Would you like some ideas for small things that might help your mood?",
        "It's okay to take things one step at a time.",
        "You matter, even if it doesn't feel like it right now.",
        "Is there anything that usually brings you comfort?"
    ],
    "adhd": [
        "Many people with ADHD feel that way. Would you like to talk about coping strategies?",
        "ADHD can make things challenging. What are you struggling with most right now?",
        "You're not alone. Are there any strategies that have helped you before?",
        "Would you like tips on focus or organization?",
        "Sometimes breaking tasks into smaller steps can help. Want to try that?",
        "Would reminders or routines be helpful for you?",
        "It's okay to ask for help with organization.",
        "Do you want to talk about ways to manage distractions?",
        "Would you like to share more about your experience with ADHD?"
    ],
    "goodbye": [
        "Take care of yourself. I'm here whenever you need support.",
        "Goodbye! Remember, you're not alone.",
        "See you next time. Wishing you well.",
        "Take care! Reach out whenever you need to talk.",
        "Bye for now! Take care of yourself.",
        "Hope to talk again soon. Stay safe.",
        "Wishing you a peaceful day. Goodbye!",
        "Remember, support is always here if you need it."
    ]
}

# List of phrases that indicate a mental health crisis.
crisis_keywords = [
    "kill myself", "want to die", "suicide", "end it all", "hurt myself",
    "hurt someone", "kill someone", "can't go on", "no reason to live",
    "ending my life", "take my own life", "can't take it anymore", "give up on life",
    "wish I was dead", "wish I weren't alive", "life is pointless", "I want to disappear",
    "I want to end everything", "I want to hurt myself", "I want to hurt others",
    "I feel like dying", "I feel like ending it", "I can't do this anymore"
]