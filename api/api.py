from fastapi import FastAPI, HTTPException
import pickle
import uvicorn
import pandas as pd

from functions.functions import (
    recomendar_hibrido,
    avaliar_modelo,
    avaliar_modelo_parallel,
)

app = FastAPI()

# Carregar os modelos salvos
with open("model/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

user_path = "model/files/usuarios.csv"
site_path = "model/files/paginas.csv"

paginas = pd.read_csv(site_path)

usuarios = pd.read_csv(user_path)


@app.get("/recomendar/{user_id}")
def recomendar(user_id: str):
    # try:
    recomendacoes = recomendar_hibrido(
        user_id, usuarios, paginas, kmeans, vectorizer, tfidf_matrix
    )
    return {"recomendacoes": list(recomendacoes)}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


@app.get("/avaliar_modelo")
def avaliar():
    validacao_path = "datathon_files/validacao.csv"
    validacao = pd.read_csv(validacao_path)
    try:
        precisao = avaliar_modelo(
            validacao, usuarios, paginas, kmeans, vectorizer, tfidf_matrix
        )
        return {"precisao": precisao}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Executando a API diretamente com uvicorn.run()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
