import json

output = "pull_commit.json"
dataset = "repositories_java.json"

# Carica i dati dei repository dal file JSON
with open(dataset, 'r') as f:
    repositories = json.load(f)

pul_com = []

for repo in repositories:

    name_rep = repo.get("name")
    commit_url = repo.get("commits_url", "").replace("{/sha}", "")
    pull_url = repo.get("pulls_url", "").replace("{/number}", "")
    pul_com.append({"name": name_rep, "commits_url": commit_url, "pulls_url": pull_url})

# Salva i dati in un file JSON
with open(output, 'w') as p:
    json.dump(pul_com, p, indent=4)

print(f"Dati salvati con successo in {output}")
