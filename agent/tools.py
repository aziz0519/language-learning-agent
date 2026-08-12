import os
import json
import random

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

translation_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)

@tool
def get_n_random_words(language: str, n: int) -> list:
    """
    Selects a specified number of random words from a language-specific word list.

    """
    path = os.path.join("data", f"{language}", "word-list-cleaned.json")

    with open(path) as f:
        word_list = json.load(f)

