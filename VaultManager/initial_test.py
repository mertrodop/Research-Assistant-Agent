from langchain_openrouter import ChatOpenRouter

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

llm = ChatOpenRouter(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=2048,
)

response = llm.invoke(
    "Who are you?"
)

print(response.content)