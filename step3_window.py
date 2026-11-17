from google.genai import types
from config import client, MODEL

WINDOW = 3   # непарне число: вікно має починатися з повідомлення user

history = []

def ask(text: str) -> str:
    history.append(types.Content(role="user", parts=[types.Part(text=text)]))
    window = history[-WINDOW:]
    response = client.models.generate_content(model=MODEL, contents=window)
    history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
    print(f"[у контексті {len(window)} з {len(history) - 1} повідомлень]")
    return response.text

messages = [
    "Привіт! Мене звати Олена.",
    "Я шукаю ноутбук для роботи.",
    "Бюджет до 40 тисяч гривень.",
    "Як мене звати?",
]

for m in messages:
    print("USER:", m)
    print("MODEL:", ask(m), "\n")
