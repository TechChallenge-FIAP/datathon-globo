# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import glob

user_path = "datathon_files/files/treino"
site_path = "datathon_files/itens/itens"

site_files = glob.glob(os.path.join(site_path, "*.csv"))
paginas = pd.concat((pd.read_csv(f) for f in site_files), ignore_index=True)

user_files = glob.glob(os.path.join(user_path, "*.csv"))
usuarios = pd.concat((pd.read_csv(f) for f in user_files), ignore_index=True)

# Conversão dos IDs das páginas visitadas em vetores TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limitando o número de features
history_vectors = vectorizer.fit_transform(usuarios["history"])

# Preparando os dados para o KMeans
X = history_vectors

# Definindo o número de clusters
num_clusters = 10

# Criando o modelo KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Atribuindo os clusters aos usuários
usuarios["cluster"] = kmeans.labels_

# Salvando o modelo KMeans
with open("model/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Salvando o vetor TF-IDF
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Combinando informações relevantes das notícias
paginas["conteudo"] = (
    paginas["title"] + " " + paginas["body"] + " " + paginas["caption"]
).fillna("")

# Criando a matriz TF-IDF para as páginas
tfidf_matrix = vectorizer.fit_transform(paginas["conteudo"])

# Salvando a matriz TF-IDF
with open("model/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)


usuarios.to_csv("model/files/usuarios.csv", index=False)
paginas.to_csv("model/files/paginas.csv", index=False)
