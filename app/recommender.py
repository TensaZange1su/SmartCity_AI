import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Основные модели ===
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Intent-категории ===
INTENT_CATEGORY_MAP = {
    "night_activity": ["Бар", "Паб", "Караоке", "Клуб", "Развлекательное заведение", "Места отдыха / Развлекательные заведения"],
    "entertainment": ["Кинотеатр", "Квест", "Боулинг", "Театр", "Музей", "Парк", "Развлечения"],
    "nature_relax": ["Парк", "Набережная", "Озеро", "Природный объект", "Туризм / Отдых"],
    "food_drink": ["Кафе", "Ресторан", "Чайхана", "Столовая", "Фастфуд", "Пицца", "Общественное питание"],
    "kids_family": ["Детский сад", "Центр развития", "Семейный центр", "Школа", "Игровая площадка"],
    "health_wellness": ["Спа", "Салон красоты", "Массаж", "Фитнес", "Йога", "Красота / Здоровье"],
    "shopping": ["Магазин", "Супермаркет", "Бутик", "Торговый центр"],
    "tourism": ["Отель", "Хостел", "Гостиница", "Туризм / Отдых"]
}

# === Сценарии пользователя ===
SCENARIO_KEYWORDS = {
    "Парк с природой для отдыха": ["парк", "природа", "озеро", "река", "сад", "зелень", "прогулка", "отдых", "набережная"],
    "Активный день": ["спорт", "тренировка", "активность", "поход", "скейт", "баскетбол", "парк", "аттракцион"],
    "Семейный день": ["дет", "семья", "развлечения", "игровая", "зоопарк", "центр", "развитие"],
    "Культурный выходной": ["музей", "театр", "галерея", "кинотеатр", "история", "экскурсия"],
    "Ночная жизнь": ["бар", "караоке", "паб", "ночной клуб", "вечеринка", "друзья", "развлечения"],
    "Еда и напитки": ["кафе", "ресторан", "чайхана", "фастфуд", "еда", "кофе", "пицца", "десерт"]
}


# === Определение намерения пользователя ===
def classify_intent(user_query: str) -> str:
    labels = list(INTENT_CATEGORY_MAP.keys())
    result = intent_classifier(user_query, candidate_labels=labels)
    return result["labels"][0] if result["labels"] else "general"


# === Фильтрация по намерению ===
def filter_by_intent(df, intent):
    if intent not in INTENT_CATEGORY_MAP:
        return df
    allowed = INTENT_CATEGORY_MAP[intent]
    mask = df["category"].apply(lambda x: any(cat.lower() in str(x).lower() for cat in allowed))
    filtered = df[mask]
    return filtered if not filtered.empty else df


# === Сценарное взвешивание ===
def apply_scenario_weighting(df, user_type):
    """Увеличивает оценку схожести, если категория совпадает с ключевыми словами сценария."""
    if user_type not in SCENARIO_KEYWORDS:
        return df

    keywords = SCENARIO_KEYWORDS[user_type]
    df["scenario_boost"] = df["category"].apply(
        lambda x: 1.2 if any(k.lower() in str(x).lower() for k in keywords) else 1.0
    )
    df["similarity"] = df["similarity"] * df["scenario_boost"]
    return df


# === Семантический поиск ===
def recommend_places(query, user_lat, user_lon, df, model, index, radius_km=5, top_k=10):
    from sklearn.metrics.pairwise import cosine_similarity

    query_vector = model.encode([query])
    vectors = df["embedding"].tolist()
    similarities = cosine_similarity(query_vector, vectors)[0]
    df["similarity"] = similarities

    # Фильтрация по радиусу
    df["dist_km"] = np.sqrt((df["latitude"] - user_lat) ** 2 + (df["longitude"] - user_lon) ** 2) * 111
    nearby = df[df["dist_km"] <= radius_km].copy()

    top_results = nearby.sort_values("similarity", ascending=False).head(top_k)
    return top_results


# === Реранкинг ===
def rerank_results(query, results_df):
    pairs = [(query, row["description"]) for _, row in results_df.iterrows()]
    scores = cross_encoder.predict(pairs)
    results_df["rerank_score"] = scores
    return results_df.sort_values("rerank_score", ascending=False)


# === LLM фильтрация ===
def llm_filter(query, user_type, results_df):
    scenario_context = f"Сценарий пользователя: {user_type}. "
    if user_type in SCENARIO_KEYWORDS:
        scenario_context += "Ключевые темы: " + ", ".join(SCENARIO_KEYWORDS[user_type]) + "."

    context = "\n".join([
        f"{row['name']} — {row.get('description', 'Без описания')} "
        f"(категория: {row.get('category', 'Неизвестно')}, {row.get('dist_km', 0):.1f} км)"
        for _, row in results_df.iterrows()
    ])

    prompt = (
        f"Ты — умный городской помощник по Астане.\n"
        f"Запрос: '{query}'\n"
        f"{scenario_context}\n\n"
        f"Проанализируй список ниже и выбери только те места, которые лучше всего соответствуют сценарию и запросу.\n"
        f"Список мест:\n{context}\n\n"
        f"Ответь только списком названий через запятую без пояснений."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        names = [n.strip() for n in content.split(",") if n.strip()]
        filtered_df = results_df[results_df["name"].isin(names)]
        return filtered_df if not filtered_df.empty else results_df
    except Exception as e:
        print(f"⚠ Ошибка LLM-фильтра: {e}")
        return results_df


# === Эмодзи и объяснения ===
def explain_recommendations(results_df, user_type):
    explanations = []
    for _, row in results_df.iterrows():
        cat = (row.get("category", "") + " " + row.get("subcategory", "") + " " + row.get("description", "")).lower()

        if any(k in cat for k in ["кафе", "ресторан", "бар", "пицца", "еда", "кофе", "чайхана", "фастфуд"]):
            emoji = "☕"
        elif any(k in cat for k in ["ночн", "паб", "вечерин", "караоке", "дискотек", "развлеч"]):
            emoji = "🌙"
        elif any(k in cat for k in ["парк", "сад", "природ", "озеро", "отдых", "туризм"]):
            emoji = "🌿"
        elif any(k in cat for k in ["дет", "школ", "семей", "развит", "игров"]):
            emoji = "👶"
        elif any(k in cat for k in ["магазин", "торгов", "рынок", "бутик", "супермаркет"]):
            emoji = "🛍️"
        elif any(k in cat for k in ["спорт", "фитнес", "йога", "плаван", "тренажер"]):
            emoji = "🏀"
        elif any(k in cat for k in ["массаж", "спа", "красота", "здоровье"]):
            emoji = "💆"
        else:
            emoji = "📍"

        explanations.append(
            f"{row['name']} — {row.get('category', 'Без категории')} "
            f"({row['dist_km']:.2f} км от вас) {emoji} — интересное место поблизости."
        )

    return explanations


# === Главная функция ===
def generate_smart_recommendations(query, user_lat, user_lon, df, model, index, user_type, radius_km=5, top_k=10):
    intent = classify_intent(query)
    df_filtered = filter_by_intent(df, intent)

    base_results = recommend_places(query, user_lat, user_lon, df_filtered, model, index, radius_km, top_k * 2)
    base_results = apply_scenario_weighting(base_results, user_type)  # <— добавлено

    reranked = rerank_results(query, base_results)
    filtered = llm_filter(query, user_type, reranked.head(top_k))

    return filtered.head(top_k)
