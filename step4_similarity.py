import numpy as np
from google.genai import types
from config import client, EMBED_MODEL, EMBED_DIM

sentences = [
    "Клієнт їздить на Toyota Camry",          # S1
    "Яке авто в клієнта?",                    # S2
    "What car does the customer drive?",      # S3
    "Клієнтка — вегетаріанка",                # S4
    "Порадь, що приготувати на вечерю",       # S5
    "Оплата лише безготівковим розрахунком",  # S6
    "Чи можна заплатити карткою?",            # S7
    "Сьогодні в Києві сонячно",               # S8
]

result = client.models.embed_content(
    model=EMBED_MODEL,
    contents=sentences,
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=EMBED_DIM,
    ),
)

vectors = np.array([e.values for e in result.embeddings])
print("Форма масиву:", vectors.shape)      # очікується (8, 768)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

n = len(sentences)
print("\n      " + "  ".join(f"S{j+1:<3}" for j in range(n)))
for i in range(n):
    row = "  ".join(f"{cosine(vectors[i], vectors[j]):.2f}" for j in range(n))
    print(f"S{i+1:<3}  {row}")

pairs = [(cosine(vectors[i], vectors[j]), i, j)
         for i in range(n) for j in range(i + 1, n)]
pairs.sort(reverse=True)
print("\nТоп-3 найсхожіші пари:")
for score, i, j in pairs[:3]:
    print(f"{score:.3f}  S{i+1} ↔ S{j+1}")
print("\nНайменш схожа пара:")
score, i, j = pairs[-1]
print(f"{score:.3f}  S{i+1} ↔ S{j+1}")