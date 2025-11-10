# app/vector_store.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import torch

EMBEDDINGS_FILE = "data/embeddings.npy"  # путь для сохранения embeddings


def build_vector_index(df, force_rebuild=False):
    """
    Строит FAISS индекс и возвращает модель и индекс.
    Использует CPU для SentenceTransformer.
    Сохраняет embeddings на диск для последующих запусков.
    """
    # Принудительно CPU, чтобы избежать meta tensor ошибок
    device = "cpu"

    # Отключаем предупреждение про symlinks Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Загружаем модель
    print("Загружаем SentenceTransformer модель на CPU...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Загружаем или вычисляем embeddings
    if os.path.exists(EMBEDDINGS_FILE) and not force_rebuild:
        print("Загружаем embeddings с диска...")
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        print("Вычисляем embeddings для всех POI...")
        texts = df["description"].fillna("").tolist()
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"Embeddings сохранены в {EMBEDDINGS_FILE}")

    # Добавляем embeddings в DataFrame
    df["embedding"] = list(embeddings)

    # Создаём FAISS индекс
    print("Строим FAISS индекс...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    print(f"FAISS индекс создан: {index.ntotal} векторов")
    return model, index


def search_similar(query, model, index, df, top_k=5):
    """
    Семантический поиск среди POI.
    Возвращает индексы и cosine similarity с query.
    """
    # Вектор запроса
    query_vec = model.encode([query], device="cpu")[0].astype(np.float32)

    # Cosine similarity
    embeddings = np.vstack(df["embedding"].values).astype(np.float32)
    scores = embeddings @ query_vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10)

    sorted_idx = np.argsort(-scores)
    top_idx = sorted_idx[:min(top_k, len(df))]

    return top_idx, scores[top_idx]
