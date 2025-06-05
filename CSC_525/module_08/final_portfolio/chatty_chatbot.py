import random
import openai # type: ignore
from api_key import API_KEY
from info_bits import INFO_BITS
from inputimeout import inputimeout, TimeoutOccurred # type: ignore
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
    client = openai.OpenAI(api_key=API_KEY) # Login to the OpenAI client.
    info = random.choice(INFO_BITS) # Select a random bit of information to start.
    # Prepare the opening system prompt.
    system_prompt = (
        "You are a helpful and engaging assistant. "
        "After each response, suggest two possible follow-up questions or comments the"
        "user might want to ask, "
        "formatted as '\nOption 1:' and '\nOption 2:'."
    )
    # Prepare the opening message and ask for two options.
    opening_prompt = (
        f"Here's something interesting: {info}\n"
        f"Based on the fact '{info}', what would you like to know or discuss?\n"
        "Also, suggest two possible follow-up questions or comments the user might want"
        "to ask, "
        "formatted as '\nOption 1:' and '\nOption 2:'."
    )
    # Get the assistant's opening message with options.
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": opening_prompt}
        ]
    )
    bot_reply = response.choices[0].message.content
    print("Chatbot:", bot_reply)
    # Initialize the conversation history for context.
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": bot_reply}
    ]
    # Loop to keep the conversation going until the user decides to exit.
    while True:
        try:
            user_input = inputimeout(prompt="You: ", timeout=30)
        except TimeoutOccurred:
            print("Chatbot: Are you still there?")
            try:
                user_input = inputimeout(prompt="You: ", timeout=30)
            except TimeoutOccurred:
                print("Chatbot: I'll be here if you want to continue later. Goodbye!")
                break
        # Check for exit conditions.
        if user_input.lower() in {"exit", "quit", "bye", "goodbye", "q", ""}:
            print("Chatbot: Goodbye! Have a great day!")
            break
        # Add user message to conversation history.
        conversation.append({
            "role": "user",
            "content": f"{user_input} (The context is: {info})"
        })
        # Get response from OpenAI's GPT model, including suggestions.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        bot_reply = response.choices[0].message.content
        # Display and add the assistant's reply to the conversation history.
        print("Chatbot:", bot_reply)
        conversation.append({"role": "assistant", "content": bot_reply})

# The big red activation button.
if __name__ == "__main__":
    main()