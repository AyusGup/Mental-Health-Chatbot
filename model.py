import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chatbot:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        """
        Initializes the chatbot model and tokenizer.

        Parameters:
        - model_name: Name of the model to load.
        - device: Optional device override (CPU/GPU).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

        # Auto-detect device (GPU if available)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Stores user sessions
        self.user_sessions = {}

    def get_chat_history(self, user_id):
        """
        Retrieves chat history for a user, initializing if not present.

        Parameters:
        - user_id: Unique identifier for the user.

        Returns:
        - Chat history list.
        """
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "history": [
                    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"}
                ],
                "last_active": time.time()  # Initialize last active timestamp
            }
        return self.user_sessions[user_id]["history"]

    def generate_response(self, user_id, user_input, max_tokens=256, temperature=0.7, top_k=50, top_p=0.95):
        """
        Generates a chatbot response while maintaining session history.

        Parameters:
        - user_id: Unique identifier for the user.
        - user_input: Message from the user.
        - max_tokens: Maximum number of tokens in response.
        - temperature: Sampling temperature (higher = more creative).
        - top_k: Limits sampling pool to top-k tokens.
        - top_p: Nucleus sampling probability.

        Returns:
        - The chatbot's response.
        """

        # 🔹 Clean expired sessions before proceeding
        self.remove_expired_sessions()

        # Get session history
        chat_history = self.get_chat_history(user_id)

        # Append user message
        chat_history.append({"role": "user", "content": user_input})

        # Update last active timestamp
        self.user_sessions[user_id]["last_active"] = time.time()

        print(f"Chat history for user {user_id}: {chat_history}")

        # Format input using chat template
        prompt = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(user_input)[-1].strip()

        # Append bot response to chat history
        chat_history.append({"role": "assistant", "content": response})

        return response

    def remove_expired_sessions(self, session_expiry=3600):
        """
        Removes expired user sessions based on inactivity.

        Parameters:
        - session_expiry: Time in seconds before a session is removed (default 1 hour).
        """
        current_time = time.time()

        expired_users = [user_id for user_id, session in self.user_sessions.items() 
                         if current_time - session["last_active"] > session_expiry]

        for user_id in expired_users:
            del self.user_sessions[user_id]
            print(f"Session expired for user: {user_id}")


