import random
import openai # type: ignore
from api_key import API_KEY
from info_bits import get_info_bits
from inputimeout import inputimeout, TimeoutOccurred # type: ignore
# --------------------------------------------------------------------------------------
# Chatty Chatbot
#
# A simple NLP-powered chatbot that starts a conversation with a random bit of
# information and continues chatting with the user until they choose to exit.
# Powered by OpenAI's GPT model.
#
# Usage:
#   python chatty_chatbot.py
# --------------------------------------------------------------------------------------
def get_openai_client():
    """Initialize and returns the OpenAI client."""
    client = openai.OpenAI(
        api_key=API_KEY
    )
    return client

def get_random_info():
    """Returns a random bit of information from INFO_BITS."""
    info_bits = get_info_bits()
    info = random.choice(info_bits)
    return info

def get_system_prompt():
    """Returns the system prompt for the assistant, instructing it to always suggest
    two follow-up questions or comments after each response."""
    part1 = "You are a helpful and engaging assistant. "
    part2 = "After each response, suggest two follow-up questions or comments the "
    part3 = "user might want to ask, "
    part4 = "formatted as '\n\tOption 1:' and '\n\tOption 2:'."
    system_prompt = part1+part2+part3+part4
    return system_prompt

def get_opening_prompt(info):
    """Returns the opening prompt for the assistant, including the random fact and
    instructions to suggest two follow-up options."""
    line1 = f"Here's something interesting: {info}\n"
    line2 = f"Based on the fact '{info}', what would you like to know or discuss?\n"
    line3 = "Also, suggest two possible follow-up questions or comments the user might"
    line4 = "want to ask, "
    line5 = "formatted as '\n\tOption 1:' and '\n\tOption 2:'."
    opening_prompt = line1+line2+line3+line4+line5
    return opening_prompt

def get_assistant_reply(client, system_prompt, opening_prompt):
    """Gets the assistant's opening message with options."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": opening_prompt}
        ]
    )
    assistant_reply = response.choices[0].message.content
    return assistant_reply

def get_user_input():
    """Prompts the user for input with a timeout. If the user does not respond within
    42 seconds, ask if they are still there. If there is still no response, exit."""
    time_limit = 42
    try:
        return inputimeout(prompt="You: ", timeout=time_limit)
    except TimeoutOccurred:
        print("Chatbot: Are you still there?")
        try:
            return inputimeout(prompt="You: ", timeout=time_limit)
        except TimeoutOccurred:
            print("Chatbot: I'll be here if you want to continue later. Goodbye!")
            return None

def get_chatbot_response(client, conversation):
    """Gets the chatbot's response from OpenAI and return the reply, given the conversation history."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    chatbot_reply = response.choices[0].message.content
    return chatbot_reply

def main():
    """Main driver function. Sets up the conversation, prints the opening message,
    and continues the chat loop until the user exits or times out."""
    client = get_openai_client()
    info = get_random_info()
    system_prompt = get_system_prompt()
    opening_prompt = get_opening_prompt(info)
    bot_reply = get_assistant_reply(client, system_prompt, opening_prompt)
    print("Chatbot:", bot_reply)
    # Initialize conversation history with system and assistant messages.
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": bot_reply}
    ]
    while True:
        user_input = get_user_input()
        if user_input is None:
            break
        # Check for exit commands.
        if user_input.lower() in {"exit","quit","bye","goodbye","q","leave"}:
            print("Chatbot: Goodbye! Have a great day!")
            break
        # Add user message to conversation history.
        conversation.append({
            "role": "user",
            "content": f"{user_input} (The context is: {info})"
        })
        # Get and print chatbot response, then add to conversation history.
        bot_reply = get_chatbot_response(client, conversation)
        print("Chatbot:", bot_reply)
        # Append the chatbot's reply to conversation history.
        conversation.append({"role": "assistant", "content": bot_reply})

# The big red activation button.
if __name__ == "__main__":
    main() # Running the main driver function.