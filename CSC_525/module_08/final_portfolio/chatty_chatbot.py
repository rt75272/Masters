import random
import openai # type: ignore
from api_key import API_KEY
from info_bits import INFO_BITS
# --------------------------------------------------------------------------------------
# Chatty Chatbot
#
# A simple NLP-powered chatbot that starts a conversation with a random bit of
# information and continues chatting with the user until they choose to exit. Powered
# by OpenAI's GPT model.
#
# Usage:
#   python chatty_chatbot.py
# --------------------------------------------------------------------------------------
def main():
    """Main driver function. Starts with a random bit of information and continues the
    conversation until the user types 'exit', 'quit', or 'bye'."""
    # Login to the OpenAI client.
    client = openai.OpenAI(api_key=API_KEY)
    # Select a random bit of information to start the conversation.
    info = random.choice(INFO_BITS)
    print(f"Here's something interesting: {info}")
    print(f"Based on the fact '{info}', what would you like to know or discuss?")
    # Initialize the conversation history for context.
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful and engaging assistant."
        },
        {
            "role": "assistant",
            "content": f"Here's something interesting: {info}"
        }
    ]
    # Loop to keep the conversation going until the user decides to exit.
    while True:
        user_input = input("You: ")
        # Check for exit conditions.
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Chatbot: Goodbye! Have a great day!")
            break
        # Add user message to conversation history.
        conversation.append({
            "role": "user",
            "content": f"{user_input} (The context is: {info})"
        })
        # Get response from OpenAI's GPT model.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        bot_reply = response.choices[0].message.content
        # Print and add the assistant's reply to the conversation history.
        print("Chatbot:", bot_reply)
        conversation.append({"role": "assistant", "content": bot_reply})

# The big red activation button.
if __name__ == "__main__":
    main()
