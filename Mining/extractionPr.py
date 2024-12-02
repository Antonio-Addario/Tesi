import json
import requests
from github import Github, RateLimitExceededException, GithubException
from pymongo.errors import DocumentTooLarge
from pymongo.mongo_client import MongoClient

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']
projects = db['projects']
pull_requests_collection = db['pull_requests_new']
skipped_collection = db['repository_saltati']  # Collezione per repository saltati

# Connessione a GitHub tramite PyGithub
token = "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"
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}
g = Github(token)

# URL delle pull request
repo_file = "fullNameRep.json"
with open(repo_file, 'r') as f:
    repository_list = json.load(f)

for name in repository_list:
    print(f"Elaborazione del repository: {name}")

    # Controlla se il repository è già stato elaborato
    if projects.find_one({"repository_name": name}):
        print(f"Repository {name} già presente. Saltando...")
        continue

    try:
        projectData = {
            'repository_name': name
        }
        project_result = projects.insert_one(projectData)
        project_id = project_result.inserted_id
        repo = g.get_repo(name)
        pull_requests = repo.get_pulls(state="all")
        numeroPR = pull_requests.totalCount

        # Controlla se c'è una pull request già elaborata per questo repository
        last_pr = pull_requests_collection.find_one({"repository_id": project_id}, sort=[("created_at", -1)])
        resume_from_pr = None
        if last_pr:
            resume_from_pr = last_pr.get("created_at")

        for pr in pull_requests:
            # Se è stato trovato un PR da cui riprendere, saltare quelli già elaborati
            if resume_from_pr and pr.created_at <= resume_from_pr:
                numeroPR -= 1
                continue

            # Ottieni l'URL del diff
            diff_url = pr.diff_url
            diff_info = ""

            # Ottieni il contenuto del diff
            try:
                response = requests.get(diff_url, headers=headers)
                response.raise_for_status()
                diff_info = response.text
            except requests.exceptions.RequestException as e:
                print(f"Errore nel recupero del diff per PR {pr.title}: {e}")

            # Ottieni il messaggio di commit
            commit_message = "N/A"
            try:
                commit = repo.get_commit(pr.head.sha)
                commit_message = commit.commit.message
            except GithubException as e:
                print(f"Errore nel recupero del messaggio di commit per PR {pr.title}: {e}")

            # Aggiungi informazioni sulle issue chiuse
            closed_issue = None
            try:
                if pr.issue_url:
                    issue_number = pr.number
                    issue = repo.get_issue(number=issue_number)
                    if issue.state == "closed":
                        closed_issue = {
                            'issue_number': issue.number,
                            'title': issue.title,
                            'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                            'comments': [comment.body for comment in issue.get_comments()]
                        }
            except GithubException as e:
                print(f"Errore nel recupero dell'issue per PR {pr.title}: {e}")

            pull_request_data = {
                "repository_id": project_id,
                "title": pr.title,
                "body_message": pr.body or "",
                "commit_message": commit_message,
                "diff": diff_info,
                "issue": closed_issue,
                "created_at": pr.created_at.isoformat()
            }

            try:
                pull_requests_collection.insert_one(pull_request_data)
                print(f"Inserita PR {pr.title} per il repository {name}")
            except DocumentTooLarge:
                # Se il documento è troppo grande, rimuovi il campo diff e riprova
                print(f"Il documento per la PR '{pr.title}' è troppo grande. Rimuovendo il diff.")
                pull_request_data.pop("diff")
                pull_requests_collection.insert_one(pull_request_data)
                print(f"Inserita PR {pr.title} senza diff per il repository {name}")

            numeroPR -= 1
            print(f"Rimangono da inserire {numeroPR} PR per il repository {name}")

    except RateLimitExceededException:
        print(f"Raggiunto il limite di richieste per il repository {name}. Riprovare più tardi.")
        break
    except GithubException as e:
        print(f"Errore generico durante l'elaborazione del repository {name}: {e}")

print("Tutte le informazioni delle PR sono state salvate in MongoDB.")
