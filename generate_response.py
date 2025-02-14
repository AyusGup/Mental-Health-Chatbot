import time
import torch
from transformers import pipeline


def chat_with_user(pipe, user_sessions, user_id, user_input):
    """
    Generates a chatbot response for a given user while maintaining session history.

    Parameters:
    - pipe: The text-generation model pipeline.
    - user_sessions: Dictionary storing chat history for each user.
    - user_id: The unique identifier for the user.
    - user_input: The message from the user.

    Returns:
    - The chatbot's response.
    """
    # Initialize chat history if user is new
    if user_id not in user_sessions:
        user_sessions[user_id] = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"}
        ]

    # Add user message to history
    user_sessions[user_id].append({"role": "user", "content": user_input})

    # Format conversation with the chat template
    prompt = pipe.tokenizer.apply_chat_template(
        user_sessions[user_id], tokenize=False, add_generation_prompt=True
    )

    # Generate response
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # Extract response text
    response = outputs[0]["generated_text"].split(user_input)[-1].strip()

    # Add response to history
    user_sessions[user_id].append({"role": "assistant", "content": response})

    return response

def remove_expired_sessions(user_sessions, session_expiry=3600):
    """
    Removes user sessions that have been inactive for more than the expiry time (default: 1 hour).

    Parameters:
    - user_sessions: Dictionary storing chat history for each user along with their last activity timestamp.
    - session_expiry: Expiry time in seconds (default: 3600s = 1 hour).

    Returns:
    - None (modifies user_sessions in place).
    """
    current_time = time.time()
    expired_users = [user_id for user_id, data in user_sessions.items() if current_time - data["last_active"] > session_expiry]

    for user_id in expired_users:
        del user_sessions[user_id]
        print(f"Session expired for user: {user_id}")
