import pandas as pd
from toolkit.load import clean_data

def group_by_num(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include='number').dropna(axis=1, how="all")
    if num.empty:
        raise ValueError('no numeric columns in dataset')
    return num

def group_by_house(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if 'Hogwarts House' not in df.columns:
        raise ValueError('no Hogwarts House column in dataset')
    data = {}
    houses = sorted(df['Hogwarts House'].dropna().unique())
    if not houses:
        raise ValueError('no house labels in dataset')
    for house in houses:
        data[house] = df[df['Hogwarts House'] == house]
    return data

def group_by_course(groups: dict[str, pd.DataFrame],
                    course: str) -> dict[str, list[float]]:
    data = {}
    for house, students in groups.items():
        data[house] = clean_data(students[course])
    return data
