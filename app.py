# app.py

from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Carregar modelos e dados
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("nn_model.pkl", "rb") as f:
    nn_model = pickle.load(f)
site_df = pd.read_pickle("site_with_text.pkl")
user_df = pd.read_pickle("preprocessed_users.pkl")

# Mapear páginas para índices
indices = pd.Series(site_df.index, index=site_df["page"]).drop_duplicates()


def get_recommendations(user_history):
    # Obter índices das notícias no histórico do usuário
    idxs = [indices.get(page) for page in user_history if indices.get(page) is not None]

    # Se não houver histórico válido, retornar notícias recentes
    if not idxs:
        recent_news = site_df.sort_values("issued", ascending=False).head(10)
        return recent_news["page"].tolist()

    # Construir o perfil do usuário pela média das representações TF-IDF das notícias lidas
    user_tfidf = tfidf.transform(site_df.loc[idxs]["text"])

    # Verificar se user_tfidf tem mais de uma amostra
    if user_tfidf.shape[0] == 1:
        user_profile = user_tfidf.toarray()
    else:
        user_profile = user_tfidf.mean(axis=0)

        # Converter para array numpy
        if isinstance(user_profile, np.matrix):
            user_profile = np.asarray(user_profile)
        else:
            user_profile = user_profile.A  # Converte matriz esparsa para array

    # Garantir que user_profile seja 2D
    user_profile = np.array(user_profile).reshape(1, -1)

    # Encontrar as notícias mais similares ao perfil do usuário
    N_RECOMMENDATIONS = 100  # Obter mais recomendações para filtragem posterior
    distances, indices_nn = nn_model.kneighbors(
        user_profile, n_neighbors=N_RECOMMENDATIONS
    )

    # Obter as recomendações
    recommended = []
    for idx in indices_nn[0]:
        page_id = site_df.iloc[idx]["page"]
        # Evitar recomendar notícias já lidas
        if page_id not in user_history:
            issued_date = site_df.iloc[idx]["issued"]
            recommended.append((page_id, issued_date))
        if len(recommended) >= 100:
            break

    # Ordenar as recomendações por data de publicação (mais recentes primeiro)
    recommended.sort(key=lambda x: x[1], reverse=True)

    # Obter as 10 principais recomendações
    final_recommendations = [rec[0] for rec in recommended[:10]]

    return final_recommendations


@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("userId")
    user = user_df[user_df["userId"] == user_id]

    if user.empty:
        return jsonify({"error": "Usuário não encontrado"}), 404

    user_history = user.iloc[0]["history"]
    recommendations = get_recommendations(user_history)

    return jsonify({"userId": user_id, "recommendations": recommendations})


if __name__ == "__main__":
    app.run(debug=True)
