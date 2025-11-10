import pandas as pd

def load_poi_data(path: str = "data/gis_clean.parquet"):
    df = pd.read_parquet(path)

    # === Переименование ===
    df = df.rename(columns={
        "Название": "name",
        "Широта": "latitude",
        "Долгота": "longitude",
        "Рубрика": "category",
        "Подрубрика": "subcategory",
        "Адрес": "address",
        "Регион": "region",
        "Город": "city"
    })

    # === Очистка ===
    df = df.drop_duplicates(subset=["name", "latitude", "longitude"], keep="first")
    df = df.dropna(subset=["name", "latitude", "longitude"])

    # === Контекстное описание (для улучшения семантики) ===
    def enrich_category_text(cat):
        mapping = {
            "Общественное питание": "поесть, отдохнуть, выпить кофе или напитки",
            "Места отдыха / Развлекательные заведения": "веселиться, отдыхать, танцевать, петь, проводить время с друзьями",
            "Туризм / Отдых": "погулять, насладиться природой, расслабиться",
            "Общее образование": "учиться, развиваться, обучаться",
            "Медицинские учреждения": "получить медицинскую помощь или консультацию",
            "Красота / Здоровье": "сделать уход, массаж, расслабиться, заняться собой",
            "Магазины": "купить одежду, продукты, товары, шопинг",
        }
        return mapping.get(cat, "посетить по своим делам")

    df["description"] = (
        df["name"].astype(str)
        + ". Это место категории: " + df["category"].astype(str)
        + ". Подкатегория: " + df["subcategory"].astype(str)
        + ". Здесь можно: " + df["category"].apply(enrich_category_text)
        + ". Адрес: " + df["address"].astype(str)
    )

    return df
