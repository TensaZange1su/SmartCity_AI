    import logging
import os
import networkx as nx
import osmnx as ox


def get_local_route(start_lat, start_lon, end_lat, end_lon, dist_m=5000, network_type='walk'):
    """
    Строит локальный маршрут по графу OSM вокруг точки старта.

    Параметры:
    - start_lat, start_lon: координаты старта
    - end_lat, end_lon: координаты назначения
    - dist_m: радиус загрузки графа (в метрах)
    - network_type: 'walk', 'drive', 'bike'

    Возвращает:
    - список координат маршрута [(lat, lon), ...] или None
    """
    try:
        # Загружаем граф OSM
        logging.info(f"[routes.py] Загружаем граф OSM вокруг ({start_lat}, {start_lon})...")
        G = ox.graph_from_point((start_lat, start_lon), dist=dist_m, network_type=network_type)

        # Находим ближайшие узлы
        orig_node = ox.nearest_nodes(G, start_lon, start_lat)
        dest_node = ox.nearest_nodes(G, end_lon, end_lat)

        # Строим кратчайший путь по длине
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')

        # Преобразуем узлы в координаты
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        logging.info(f"[routes.py] ✅ Маршрут построен: {len(route_coords)} точек")
        return route_coords

    except Exception as e:
        logging.warning(f"[routes.py] ⚠ Ошибка построения локального маршрута: {e}")
        return None
