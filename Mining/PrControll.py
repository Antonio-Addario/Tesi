import json
import requests

token = "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"
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}
listaCompleta = 'pulls_commits.json'
with open(listaCompleta, 'r') as f:
    output = json.load(f)
contatore_rep = 0
for repo in output:
    response = requests.get(repo['pulls_url'], headers=headers)
    if response.status_code == 200:
        data = response.json()
        contatore = 0
        for pr in data:
            contatore += 1
        if contatore >= 20:
            contatore_rep += 1
            continue
    else:
        print("errore")

print("numero di repository con PR >20: ", contatore_rep)
