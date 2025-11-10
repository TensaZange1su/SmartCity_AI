import pandas as pd
import numpy as np
from datetime import datetime

# === 1. Загрузка ===
df = pd.read_excel("gis.xlsx")
print(f"Исходное количество строк: {len(df):,}")

# === 2. Очистка базовая ===
dup_count = df.duplicated(subset=["Id"]).sum()
df = df.drop_duplicates(subset=["Id"])
print(f"Удалено дубликатов по Id: {dup_count:,}")

# Убираем пробелы и пустые строки
df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
df = df.replace(r'^\s*$', np.nan, regex=True)

# === 3. Очистка координат ===
coord_before = len(df)
df = df[(df["Широта"].between(40, 56)) & (df["Долгота"].between(46, 88))]
df = df.dropna(subset=["Широта", "Долгота"])
coord_after = len(df)
print(f"Удалено строк с некорректными координатами: {coord_before - coord_after:,}")

# === 4. Нормализация текстовых полей ===
for col in ["Регион", "Район", "Город", "Район города"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(" область", "", regex=False)
        df[col] = df[col].str.replace(" район", "", regex=False)
        df[col] = df[col].str.strip().str.title()

# === 5. Контакты ===
if "Телефон" in df.columns:
    df["Телефон"] = df["Телефон"].astype(str).str.replace(r"[^\d+]", "", regex=True)
if "Email" in df.columns:
    df["Email"] = df["Email"].str.lower().where(df["Email"].str.match(r'^[^@]+@[^@]+\.[^@]+$'), np.nan)

# === 6. Соцсети: булевы флаги (1 — указано, 0 — нет) ===
social_cols = [
    "whatsapp", "viber", "telegram", "facebook", "instagram",
    "vkontakte", "odnoklassniki", "youtube", "twitter", "skype",
    "icq", "googleplus", "linkedin", "pinterest"
]
for col in social_cols:
    if col in df.columns:
        df[col] = df[col].notna().astype(int)

# === 7. Сброс индексов ===
df = df.reset_index(drop=True)

# === 8. Приведение object-столбцов в безопасный формат для parquet ===
def normalize_object_column(col):
    if col.apply(lambda x: isinstance(x, (pd.Timestamp, datetime))).any():
        return col.apply(lambda x: x.isoformat() if pd.notna(x) else None)
    return col.where(col.isna(), col.astype(str))

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = normalize_object_column(df[c])

# === 9. Валидация и сохранение ===
print(f"Итоговое количество строк: {len(df):,}")
print(f"Удалено всего строк: {len(pd.read_excel('gis.xlsx')) - len(df):,}")

print("\nСохранение файлов...")
df.to_csv("gis_clean.csv", index=False, encoding="utf-8")
df.to_parquet("gis_clean.parquet", index=False, engine="pyarrow")
print("Готово ✅ Файлы сохранены: gis_clean.csv и gis_clean.parquet")
