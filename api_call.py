import requests

# URL da API
url = "http://0.0.0.0:8000/recomendar/{user_id}"

# ID do usuário para o qual queremos recomendações
user_id = "f98d1132f60d46883ce49583257104d15ce723b3bbda2147c1e31ac76f0bf069"  # Substitua pelo ID do usuário desejado

# Fazendo a chamada para a API
response = requests.get(url.format(user_id=user_id))

print(response)

# Verificando se a chamada foi bem-sucedida
if response.status_code == 200:
    recomendacoes = response.json().get("recomendacoes", [])
    if recomendacoes:
        print("Recomendações para o usuário {}:".format(user_id))
        for recomendacao in recomendacoes:
            print(recomendacao)
    else:
        print("Não há recomendações disponíveis para este usuário.")
else:
    print("Erro ao chamar a API: {}".format(response.status_code))
