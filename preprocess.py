import pandas as pd


def read_crew_data():
    df = pd.read_csv("data/crew-data.csv")
    return df


def preprocess_data():
    df = read_crew_data()

    df['Message'] = df['Message'].str.lower()
    print(df['Message'])


preprocess_data()