# model_training.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import nltk
from nltk.corpus import stopwords


def train_content_based_model():
    # Carregar dados pré-processados
    site_df = pd.read_pickle("preprocessed_site.pkl")

    # Unir título, corpo e subtítulo em uma única string
    site_df["text"] = (
        site_df["title"].fillna("")
        + " "
        + site_df["body"].fillna("")
        + " "
        + site_df["caption"].fillna("")
    )

    # Baixar as stop words em português (se necessário)
    nltk.download("stopwords")
    stop_words = stopwords.words("portuguese")

    # Vectorização TF-IDF
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf.fit_transform(site_df["text"])

    # Treinar o modelo de Vizinhos Mais Próximos
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    # Salvar objetos necessários
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("nn_model.pkl", "wb") as f:
        pickle.dump(nn_model, f)
    site_df.to_pickle("site_with_text.pkl")


if __name__ == "__main__":
    train_content_based_model()
