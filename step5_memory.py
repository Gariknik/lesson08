import numpy as np
from google.genai import types
from config import client, MODEL, EMBED_MODEL, EMBED_DIM

# 1. ЗБЕРЕГТИ: факти про користувача (довгострокова пам'ять)
facts = [
    "Олена — вегетаріанка",
    "Олена живе у Львові",
    "Олена їздить на Toyota Camry 2018 року",
    "Олена працює аналітиком даних",
    "Олена не любить гострої їжі",
    "Олена планує відпустку в Карпатах у жовтні",
]

def embed(texts, task):
    r = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM),
    )
    v = np.array([e.values for e in r.embeddings])
    return v / np.linalg.norm(v, axis=1, keepdims=True)   # нормалізуємо

memory_vectors = embed(facts, "RETRIEVAL_DOCUMENT")

# 2. ЗНАЙТИ: top-k фактів, релевантних до питання
def recall(question, k=2):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    scores = memory_vectors @ q            # косинус, бо вектори нормалізовані
    top = np.argsort(scores)[::-1][:k]
    return [(facts[i], float(scores[i])) for i in top]

# 3. ПІДСТАВИТИ: знайдені факти в контекст
def answer(question):
    found = recall(question)
    print(f"\nПИТАННЯ: {question}")
    for text, score in found:
        print(f"  пам'ять: {score:.3f}  {text}")
    memory_block = "\n".join(f"- {t}" for t, _ in found)
    prompt = (
        "Відомі факти про користувача:\n"
        f"{memory_block}\n\n"
        "Відповідай коротко, враховуючи ці факти.\n"
        f"Питання: {question}"
    )
    r = client.models.generate_content(model=MODEL, contents=prompt)
    print("ВІДПОВІДЬ:", r.text)

questions = [
    "Що мені приготувати на вечерю?",
    "Яке мастило краще залити в мій двигун?",
    "Що взяти з собою в подорож?",
    "Чи подобаються мені гострі сюжети в кіно?",
]

for q in questions:
    answer(q)