# data_preprocessing.py

import pandas as pd
import os
import glob


def preprocess_users(user_df):
    # Converter strings de listas em listas reais
    user_df["history"] = user_df["history"].apply(lambda x: x.strip('"').split(", "))
    user_df["timestampHistory"] = user_df["timestampHistory"].apply(
        lambda x: [int(ts) for ts in x.strip('"').split(", ")]
    )

    # Adicionar outras conversões necessárias (se houver)
    return user_df


def preprocess_site(site_df):
    # Converter colunas de data
    site_df["Issued"] = pd.to_datetime(site_df["issued"])
    site_df["Modified"] = pd.to_datetime(site_df["modified"])

    # **Não remover notícias antigas**
    # Manter todas as notícias no dataset

    return site_df


if __name__ == "__main__":
    user_path = "datathon_files/files/treino"
    site_path = "datathon_files/itens/itens"

    site_files = glob.glob(os.path.join(site_path, "*.csv"))
    site = pd.concat((pd.read_csv(f) for f in site_files), ignore_index=True)

    user_files = glob.glob(os.path.join(user_path, "*.csv"))
    usuarios = pd.concat((pd.read_csv(f) for f in user_files), ignore_index=True)

    user_df = preprocess_users(usuarios)
    site_df = preprocess_site(site)

    # Salvar DataFrames pré-processados
    user_df.to_pickle("preprocessed_users.pkl")
    site_df.to_pickle("preprocessed_site.pkl")
