from app.data_loader import load_poi_data
from app.vector_store import build_vector_index
from app.recommender import recommend_places
from app.rag_system import generate_response

df = load_poi_data()
model, index = build_vector_index(df)

while True:
    user_query = input("\nВведите ваш запрос (или 'exit'): ")
    if user_query == "exit":
        break

    lat, lon = 51.1280, 71.4300  # пример: центр Астаны
    results = recommend_places(user_query, lat, lon, df, model, index)
    answer = generate_response(user_query, results)
    print("\nРекомендация:\n", answer)
