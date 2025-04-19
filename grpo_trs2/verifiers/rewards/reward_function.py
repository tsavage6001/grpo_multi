import openai
from typing import List, Dict
import os
import time

# Set your OpenAI API key here or via environment variable
openai.api_key = os.getenv("***")

def query_gpt(prompt: str, retries: int = 3, delay: float = 1.0) -> str:
    for _ in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            time.sleep(delay)
    return ""

def reward_function(prompts: List[List[Dict[str, str]]], completions: List[List[Dict[str, str]]], **kwargs) -> List[int]:
    """
    prompts: List of full conversations (list of message dicts)
    completions: List of completions to evaluate (one per conversation)

    Assumes `correct_diagnosis` is passed in kwargs as a list of strings.
    """
    assert "correct_diagnosis" in kwargs, "`correct_diagnosis` must be provided in kwargs"
    correct_diagnoses = kwargs["correct_diagnosis"]

    rewards = []
    for conversation, completion, true_dx in zip(prompts, completions, correct_diagnoses):
        full_convo = conversation + completion

        # Ask GPT to provide a diagnosis
        gpt_prompt = (
            "Given the following conversation between a physician and a patient, "
            "what is the most likely diagnosis? Please state only the diagnosis.\n\n"
            f"{format_conversation(full_convo)}"
        )
        model_diagnosis = query_gpt(gpt_prompt).lower()

        # Ask GPT to judge similarity
        judgment_prompt = (
            f"Does the model's diagnosis '{model_diagnosis}' mean the same thing as the correct diagnosis "
            f"'{true_dx}'? Reply with 'Yes' or 'No'."
        )
        judgment = query_gpt(judgment_prompt)

        rewards.append(1 if "yes" in judgment.lower() else 0)

    return rewards

def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format conversation messages into a readable string for GPT"""
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
