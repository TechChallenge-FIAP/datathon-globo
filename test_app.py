# test_app.py

import requests


def test_recommendation(user_id):
    url = "http://localhost:5000/recommend"
    params = {"userId": user_id}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"Recomendações para o usuário {user_id}:")
        for i, rec in enumerate(data["recommendations"], 1):
            print(f"{i}. {rec}")
    else:
        print(f"Erro: {response.status_code}")
        print(response.json())


if __name__ == "__main__":
    # Substitua pelo userId que deseja testar
    test_user_id = "e25fbee3a42d45a2914f9b061df3386b2ded2d8cc1f3d4b901419051126488b9"
    test_recommendation(test_user_id)
