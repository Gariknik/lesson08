from config import client, MODEL

print("=== Варіант А: два окремі виклики ===")
client.models.generate_content(model=MODEL, contents="Мене звати Олена. Запам'ятай.")
r = client.models.generate_content(model=MODEL, contents="Як мене звати?")
print(r.text)

print("\n=== Варіант Б: об'єкт чату ===")
chat = client.chats.create(model=MODEL)
chat.send_message("Мене звати Олена. Запам'ятай.")
r = chat.send_message("Як мене звати?")
print(r.text)
print("Повідомлень в історії:", len(chat.get_history()))