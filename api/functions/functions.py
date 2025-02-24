from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed


def recomendar_hibrido(user_id, usuarios, paginas, kmeans, vectorizer, tfidf_matrix):
    if user_id not in usuarios["userId"].values:
        return []

    user_index = usuarios[usuarios["userId"] == user_id].index[0]

    # Pegando o vetor de histórico do usuário
    user_history = usuarios.loc[user_index, "history"]
    user_vector = vectorizer.transform([user_history])

    # Identificando o cluster do usuário usando o KMeans carregado
    cluster_id = kmeans.predict(user_vector)[0]

    # Encontrar usuários no mesmo cluster
    usuarios_similares = usuarios[usuarios["cluster"] == cluster_id]

    # Coletar páginas visitadas por esses usuários
    paginas_vistas_similares = usuarios_similares["history"].explode().unique()

    # Remover páginas já visitadas pelo usuário
    paginas_vistas = set(user_history)
    paginas_recomendadas = set(paginas_vistas_similares) - paginas_vistas

    # Criar o vetor TF-IDF das páginas recomendadas
    paginas_indices = paginas[paginas["page"].isin(paginas_recomendadas)].index
    if len(paginas_indices) == 0:
        return []

    # Calcular similaridade com o vetor de histórico do usuário
    similaridades = cosine_similarity(
        user_vector, tfidf_matrix[paginas_indices]
    ).flatten()
    recomendacoes_indices = similaridades.argsort()[-5:][::-1]

    return paginas.iloc[paginas_indices[recomendacoes_indices]]["page"].values


def avaliar_modelo(
    validacao, usuarios, paginas, kmeans, vectorizer, tfidf_matrix, n_samples=100
):
    # Amostrar dados de validação
    validacao_amostra = validacao.sample(n=n_samples, random_state=42)

    acertos = 0
    total = 0

    for index, row in validacao_amostra.iterrows():
        user_id = row["userId"]
        true_history = set(row["history"])

        # Obter recomendações do modelo
        recomendacoes = recomendar_hibrido(
            user_id, usuarios, paginas, kmeans, vectorizer, tfidf_matrix
        )

        # Verificar quantas recomendações estão corretas
        recomendacoes_acertadas = true_history.intersection(set(recomendacoes))
        acertos += len(recomendacoes_acertadas)
        total += len(true_history)

        print(acertos)
        print(total)

    return acertos / total if total > 0 else 0
