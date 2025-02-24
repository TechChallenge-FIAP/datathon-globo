import requests

# URL da API
url = "http://127.0.0.1:8000/avaliar_modelo"

# Fazendo a chamada para a API
response = requests.get(url)

# Verificando se a chamada foi bem-sucedida
if response.status_code == 200:
    precisao = response.json().get("precisao", 0)
    print(f"Precisão do modelo: {precisao:.2%}")
else:
    print(f"Erro ao chamar a API: {response.status_code}")
