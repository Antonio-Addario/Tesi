import pymongo
from pymongo.mongo_client import MongoClient
import requests
import sys
from pymongo.errors import DocumentTooLarge

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']
projects_collection = db['projects']
pull_requests_collection = db['pull_requests_new']
token = "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"

# Headers per GitHub API
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

# Flag per riprendere l'esecuzione da una specifica PR
resume_from_pr = "672a3a998eb967273d911649"
found_resume_pr = False

# Itera sui progetti
for project in projects_collection.find():
    if project["repository_name"] == "emacs-lsp/lsp-mode":
        break

    # Trova tutte le pull request per il progetto corrente
    pull_requests = pull_requests_collection.find({"repository_id": project["_id"]})

    try:
        for p_r in pull_requests:
            # Verifica se bisogna riprendere l'esecuzione da una specifica PR
            if not found_resume_pr:
                if str(p_r["_id"]) == resume_from_pr:
                    found_resume_pr = True
                else:
                    continue

            url_diff = p_r.get("diff", "")

            # Controlla se il campo `diff` contiene un URL
            if url_diff.startswith("http"):
                try:
                    response = requests.get(url_diff, headers=headers)
                    response.raise_for_status()  # Solleva un'eccezione per gli errori HTTP
                    diff = response.text

                    try:
                        # Tenta di aggiornare il campo `diff` con il contenuto reale
                        pull_requests_collection.update_one(
                            {"_id": p_r["_id"]},
                            {"$set": {"diff": diff}}
                        )
                        print(f"Aggiornato il campo diff per la PR: {p_r['title']}")

                    except DocumentTooLarge:
                        print(f"Il documento per la PR '{p_r['title']}' è troppo grande. Diff saltato.")

                except requests.exceptions.RequestException as e:
                    print(f"Errore nel recupero del diff per l'URL {url_diff}: {e}. Diff saltato.")

    except pymongo.errors.CursorNotFound:
        print(f"Errore: Cursor not found per il progetto {project['repository_name']}. Progetto saltato.")

print("Aggiornamento completato.")
