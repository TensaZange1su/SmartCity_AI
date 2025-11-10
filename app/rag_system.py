import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_response(query, retrieved_df):
    """
    Генерирует текстовую рекомендацию или маршрут на основе контекста.
    Усилена смысловая фильтрация и контекст понимания.
    """
    try:
        context = "\n".join([
            f"{row['Название']} — {row.get('Описание', 'Без описания')} "
            f"(категория: {row.get('Категория', 'Неизвестно')}, {row.get('dist_km', 0):.1f} км)"
            for _, row in retrieved_df.iterrows()
        ])

        prompt = (
            f"Ты — интеллектуальный гид по городу Астана. "
            f"Пользователь задал запрос: '{query}'. "
            "Твоя задача — выбрать из предложенных мест только те, "
            "которые действительно соответствуют смыслу запроса (например, если пользователь просит парк — "
            "игнорируй магазины и кафе). "
            "Опиши в дружеском, информативном стиле 3–4 наиболее подходящих места: "
            "укажи их название, категорию, расстояние и коротко объясни, почему туда стоит сходить. "
            "Если ни одно место не подходит по смыслу — честно напиши, что поблизости нет подходящих мест.\n\n"
            f"Вот список доступных мест:\n{context}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠ Ошибка при генерации ответа: {e}"
