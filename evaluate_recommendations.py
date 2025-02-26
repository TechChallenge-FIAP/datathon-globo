# evaluate_recommendations.py

import requests
import pandas as pd
import numpy as np


def evaluate_recommendations():
    # Carregar o conjunto de validação
    validation_df = pd.read_csv("./datathon_files/validacao.csv")

    # Pré-processar o histórico
    validation_df["history"] = validation_df["history"].apply(
        lambda x: x.strip('"[]').replace("'", "").split(" ")
    )

    precisions = []
    recalls = []
    ndcgs = []

    for _, row in validation_df.iterrows():
        user_id = row["userId"]
        actual_items = set(row["history"])

        # Obter recomendações da API
        response = requests.get(
            "http://localhost:5000/recommend", params={"userId": user_id}
        )

        if response.status_code == 200:
            recommended_items = response.json()["recommendations"]
            recommended_set = set(recommended_items)

            # Cálculos
            true_positives = actual_items.intersection(recommended_set)
            precision = (
                len(true_positives) / len(recommended_items) if recommended_items else 0
            )
            recall = len(true_positives) / len(actual_items) if actual_items else 0
            precisions.append(precision)
            recalls.append(recall)

            # Calcular NDCG
            relevance = [1 if item in actual_items else 0 for item in recommended_items]
            dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = sum(
                [rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)]
            )
            ndcg = dcg / idcg if idcg != 0 else 0
            ndcgs.append(ndcg)
        else:
            print(f"Erro ao obter recomendações para o usuário {user_id}")

    # Resultados
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1_score = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall)
        else 0
    )
    avg_ndcg = np.mean(ndcgs)

    print(f"Precisão Média: {avg_precision:.4f}")
    print(f"Revocação Média: {avg_recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"NDCG Médio: {avg_ndcg:.4f}")


if __name__ == "__main__":
    evaluate_recommendations()
