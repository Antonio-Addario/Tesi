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
        # Controlla se il numero di pull request supera 10.000
        # if pull_requests.totalCount > 10000:
        #   print(f"Repository {name} ha oltre 10.000 pull request. Saltando...")
        #  skipped_collection.insert_one({"repository_name": name})  # Aggiungi il repository saltato alla collezione
        # continue

        for pr in pull_requests:
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

"""import json
import time
import sys
import requests
from github import Github, RateLimitExceededException, GithubException
from pymongo.mongo_client import MongoClient
from pymongo.errors import DocumentTooLarge

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']
collection = db['pull_requests']
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

MAX_DOCUMENT_SIZE = 16 * 1024 * 1024  # 16 MB

for name in repository_list:
    print(f"Elaborazione del repository: {name}")

    # Controlla se il repository è già stato elaborato
    if collection.find_one({"repository_name": name}):
        print(f"Repository {name} già presente. Saltando...")
        continue

    try:
        repo = g.get_repo(name)
        pull_requests = repo.get_pulls(state="all")

        # Controlla se il numero di pull request supera 10.000
        if pull_requests.totalCount > 10000:
            print(f"Repository {name} ha oltre 10.000 pull request. Saltando...")
            skipped_collection.insert_one({"repository_name": name})  # Aggiungi il repository saltato alla collezione
            continue

        all_pull_requests = []
        batch_start_date = None
        batch_end_date = None

        for pr in pull_requests:
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
                "title": pr.title,
                "body_message": pr.body or "",
                "commit_message": commit_message,
                "diff": diff_info,
                "issue": closed_issue,
                "created_at": pr.created_at.isoformat()
            }

            all_pull_requests.append(pull_request_data)

            # Aggiorna le date del batch
            if batch_start_date is None or pr.created_at < batch_start_date:
                batch_start_date = pr.created_at
            if batch_end_date is None or pr.created_at > batch_end_date:
                batch_end_date = pr.created_at

            # Stima la dimensione del documento in memoria
            estimated_size = sys.getsizeof(json.dumps({
                "repository_name": name,
                "batch_start_date": batch_start_date.isoformat() if batch_start_date else None,
                "batch_end_date": batch_end_date.isoformat() if batch_end_date else None,
                "pull_requests": all_pull_requests
            }))

            # Se la dimensione stimata supera il limite, salva il batch corrente
            if estimated_size > MAX_DOCUMENT_SIZE:
                all_pullreq_for_repo = {
                    "repository_name": name,
                    "batch_start_date": batch_start_date.isoformat() if batch_start_date else None,
                    "batch_end_date": batch_end_date.isoformat() if batch_end_date else None,
                    "pull_requests": all_pull_requests[:-1]  # Escludi l'ultimo elemento che fa superare il limite
                }
                try:
                    collection.insert_one(all_pullreq_for_repo)
                    print(f"Inserite {len(all_pull_requests) - 1} PR per il repository {name} (fino a 16 MB)")
                except DocumentTooLarge:
                    print(f"Errore: il documento per il repository {name} è troppo grande. Saltando questo batch.")

                # Inizia un nuovo batch con l'ultima PR esclusa
                all_pull_requests = [all_pull_requests[-1]]
                batch_start_date = pr.created_at
                batch_end_date = pr.created_at

        # Inserisci le rimanenti pull request se ce ne sono
        if all_pull_requests:
            all_pullreq_for_repo = {
                "repository_name": name,
                "batch_start_date": batch_start_date.isoformat() if batch_start_date else None,
                "batch_end_date": batch_end_date.isoformat() if batch_end_date else None,
                "pull_requests": all_pull_requests
            }
            try:
                collection.insert_one(all_pullreq_for_repo)
                print(f"Inserite {len(all_pull_requests)} PR rimanenti per il repository {name}")
            except DocumentTooLarge:
                print(f"Errore: il documento per il repository {name} è troppo grande. Saltando le rimanenti PR.")

    except RateLimitExceededException:
        print("Limite di richieste GitHub superato. Attendi per il reset.")
        rate_limit = g.get_rate_limit()
        reset_timestamp = rate_limit.core.reset.timestamp()
        wait_time = reset_timestamp - time.time() + 10  # Attendi fino al reset del limite
        time.sleep(wait_time)
    except GithubException as e:
        print(f"Errore generico durante l'elaborazione del repository {name}: {e}")

print("Tutte le informazioni delle PR sono state salvate in MongoDB.")"""
