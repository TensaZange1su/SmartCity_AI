import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import time

from app.data_loader import load_poi_data
from app.vector_store import build_vector_index
from app.recommender import recommend_places, explain_recommendations
from app.routes import get_local_route

# === Настройки страницы ===
st.set_page_config(page_title="SmartCity_AI", layout="wide")
st.title("🚀 SmartCity AI — RAG-гид по Астане")
st.markdown("Выберите свою позицию и интерес — система покажет ближайшие POI и маршруты до них.")

# === Геокодинг с кэшем ===
@st.cache_data
def geocode_address(address: str):
    geolocator = Nominatim(user_agent="smartcity_ai")
    try:
        location = geolocator.geocode(address, timeout=10)
        time.sleep(1)
        return location
    except Exception:
        return None

# === Загрузка данных и индекса ===
df = load_poi_data()
model, index = build_vector_index(df)

# === Инициализация session_state ===
for key in ["user_lat", "user_lon", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# === Выбор способа определения позиции ===
input_type = st.radio("Укажите вашу позицию", ["На карте", "Адрес / POI", "GPS / Автоопределение"])

# Карта по умолчанию
m = folium.Map(location=[51.1694, 71.4491], zoom_start=12)

# === Определение локации ===
if input_type == "На карте":
    st.markdown("🗺️ Кликните на карте, чтобы выбрать ваше местоположение:")
    click_data = st_folium(m, width=700, height=500)
    if click_data and click_data.get("last_clicked"):
        st.session_state.user_lat = click_data["last_clicked"]["lat"]
        st.session_state.user_lon = click_data["last_clicked"]["lng"]
        st.success(f"📍 Вы выбрали точку: {st.session_state.user_lat:.5f}, {st.session_state.user_lon:.5f}")

elif input_type == "Адрес / POI":
    location_text = st.text_input("Введите адрес или ближайший POI")
    if location_text:
        location = geocode_address(location_text)
        if location:
            st.session_state.user_lat = location.latitude
            st.session_state.user_lon = location.longitude
            st.success(f"📍 Найдено: {location.address}")
        else:
            st.error("❌ Не удалось определить координаты")

elif input_type == "GPS / Автоопределение":
    st.markdown("📡 Определяем ваше местоположение через браузер...")
    try:
        from streamlit_geolocation import st_geolocation
        user_loc = st_geolocation()
        if user_loc:
            st.session_state.user_lat = user_loc["lat"]
            st.session_state.user_lon = user_loc["lon"]
            st.success(f"📍 Ваши координаты: {st.session_state.user_lat:.5f}, {st.session_state.user_lon:.5f}")
        else:
            st.warning("⚠️ Разрешите доступ к геопозиции в браузере")
    except ImportError:
        st.error("Для работы GPS установите пакет: `pip install streamlit-geolocation`")

# === Если координаты известны ===
if st.session_state.user_lat and st.session_state.user_lon:
    folium.Marker(
        [st.session_state.user_lat, st.session_state.user_lon],
        popup="Вы здесь",
        icon=folium.Icon(color="blue", icon="user"),
    ).add_to(m)

    # === Настройки пользователя ===
    st.subheader("🎯 Настройки рекомендации")
    user_type = st.selectbox("Выберите сценарий", ["Прогулка", "Кофе/работа", "С детьми", "Турист", "Ночная активность"])
    duration_minutes = st.slider("Время прогулки (мин)", 15, 180, 60)
    user_query = st.text_input("Что вы ищете? (например: кофе, музей, парк, шопинг)")

    # --- Поиск рекомендаций ---
    if st.button("🔍 Найти рекомендации") and user_query:
        with st.spinner("Генерируем рекомендации..."):
            results = recommend_places(
                query=user_query,
                user_lat=st.session_state.user_lat,
                user_lon=st.session_state.user_lon,
                df=df,
                model=model,
                index=index,
                radius_km=5,
                top_k=5,
            )

            if results.empty:
                st.warning("Нет подходящих мест рядом 😕")
                st.session_state.results = None
            else:
                st.session_state.results = results
                st.success(f"Найдено {len(results)} мест поблизости!")

    # --- Если рекомендации уже есть ---
    if st.session_state.results is not None and not st.session_state.results.empty:
        results = st.session_state.results
        explanations = explain_recommendations(results, user_type)

        # === Центрируем карту по пользователю и найденным POI ===
        all_lats = [st.session_state.user_lat] + results["latitude"].tolist()
        all_lons = [st.session_state.user_lon] + results["longitude"].tolist()
        bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
        m.fit_bounds(bounds)

        st.subheader("📍 Рекомендации поблизости")
        for i, (_, row) in enumerate(results.iterrows()):
            st.markdown(explanations[i])

            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=f"<b>{row['name']}</b><br>{row['address']}",
                icon=folium.Icon(color="red"),
            ).add_to(m)

            route_coords = get_local_route(
                st.session_state.user_lat, st.session_state.user_lon, row["latitude"], row["longitude"]
            )
            if route_coords:
                folium.PolyLine(route_coords, color="green", weight=4, opacity=0.7).add_to(m)

# === Отображаем карту ===
st_folium(m, width=750, height=550)
