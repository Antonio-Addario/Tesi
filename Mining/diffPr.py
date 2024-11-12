import json
import requests
import time
from pymongo.mongo_client import MongoClient
from pymongo.errors import DocumentTooLarge

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']
collection = db['pull_requests']

# Token e headers per GitHub API
token = "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"
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

# Funzione per controllare il limite delle API
def check_rate_limit(headers):
    response = requests.get("https://api.github.com/rate_limit", headers=headers)
    if response.status_code == 200:
        rate_limit = response.json()['rate']['remaining']
        reset_time = response.json()['rate']['reset']
        if rate_limit < 100:
            print(f"Limite richieste quasi raggiunto. Attendi fino a {time.ctime(reset_time)}")
            wait_time = reset_time - time.time() + 10  # Aggiungi un margine di sicurezza di 10 secondi
            if wait_time > 0:
                time.sleep(wait_time)
    else:
        print(f"Errore nella richiesta del limite: {response.status_code}")

# Funzione per ottenere tutte le pull request con paginazione
def get_all_pull_requests(repo_url, headers):
    pull_requests = []
    page = 1
    per_page = 100  # Impostiamo a 100 il numero massimo di PR per pagina

    while True:
        url = f"{repo_url}?page={page}&per_page={per_page}"
        try:
            response = requests.get(url, headers=headers, timeout=10)  # Aggiungi timeout di 10 secondi
            response.raise_for_status()  # Controlla eventuali errori HTTP
        except requests.exceptions.RequestException as e:
            print(f"Errore nella richiesta delle PR: {e}. Riprovo...")
            time.sleep(2)  # Aspetta 2 secondi prima di riprovare
            continue

        prs = response.json()
        if not prs:
            break  # Esce se non ci sono più PR
        pull_requests.extend(prs)
        page += 1

    return pull_requests

# Funzione per ottenere le issue chiuse associate a una PR
def get_closed_issue(issue_url, headers):
    try:
        response = requests.get(issue_url, headers=headers, timeout=10)  # Aggiungi timeout di 10 secondi
        response.raise_for_status()
        issue = response.json()
        if issue.get('state') == "closed":
            # Gestisci i commenti associati all'issue chiusa
            comments = []
            if issue["comments"] > 0:
                comments_url = issue["comments_url"]
                comment_response = requests.get(comments_url, headers=headers, timeout=10)  # Aggiungi timeout
                if comment_response.status_code == 200:
                    comments_data = comment_response.json()
                    comments = [comment["body"] for comment in comments_data]
                else:
                    print(
                        f"Errore nel recupero dei commenti per issue {issue['number']}: {comment_response.status_code}")
            return {
                'issue_number': issue['number'],
                'title': issue['title'],
                'closed_at': issue['closed_at'],
                'comments': comments if comments else None
            }
        else:
            print(f"L'issue {issue['number']} non è chiusa")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Errore nel recupero dell'issue da {issue_url}: {e}")
    return None

# URL delle pull request
repo = "filtered_rep.json"
with open(repo, 'r') as f:
    repository = json.load(f)

for pullr in repository:
    if collection.find_one({"repository_name": pullr["name"]}):
        print(f"Repository {pullr['name']} già presente. Saltando...")
        continue

    nameRep = pullr["name"]
    pull_url = pullr["pulls_url"] + "?state=all"
    print(pull_url)
    # Controlla il limite prima di fare la richiesta
    check_rate_limit(headers)

    # Ottenere tutte le pull request con paginazione
    prs = get_all_pull_requests(pull_url, headers)
    print(f"Trovate {len(prs)} pull request per il repository {nameRep}")

    allPullreqForRepo = {
        "repository_name": nameRep,
        "pull_requests": []
    }

    for pr in prs:
        diff_url = pr["diff_url"]
        diffInfo = ""

        # Controlla il limite prima di ogni richiesta diff
        check_rate_limit(headers)

        try:
            response = requests.get(diff_url, headers=headers, timeout=10)  # Aggiungi timeout
            response.raise_for_status()
            diffInfo = response.text
        except requests.exceptions.RequestException as e:
            print(f"Errore nella richiesta del diff per PR {pr['title']}: {e}")

        commit_url = pr["commits_url"]
        commit_message = "N/A"

        # Controlla il limite prima di ogni richiesta commit
        check_rate_limit(headers)

        try:
            response = requests.get(commit_url, headers=headers, timeout=10)  # Aggiungi timeout
            response.raise_for_status()
            commitInfo = response.json()
            if commitInfo:
                commit_message = commitInfo[0]['commit']['message']
        except requests.exceptions.RequestException as e:
            print(f"Errore nella richiesta dei commit per PR {pr['title']}: {e}")

        # Controllo e recupero delle issue chiuse
        issue_url = pr.get("issue_url")
        closedIssues = None
        if issue_url:
            closedIssues = get_closed_issue(issue_url, headers)

        p_r = {
            "title": pr["title"],
            "body_message": pr["body"] or "",
            "commit_message": commit_message,
            "diff": diffInfo,
            "issue": closedIssues
        }

        allPullreqForRepo["pull_requests"].append(p_r)

    # Inserimento nel database con gestione dell'errore DocumentTooLarge
    try:
        if allPullreqForRepo["pull_requests"]:
            collection.insert_one(allPullreqForRepo)
            print(f"Inserite {len(allPullreqForRepo['pull_requests'])} PR per il repository {nameRep}")
    except DocumentTooLarge:
        print(f"Errore: il documento per il repository {nameRep} è troppo grande. Saltando questo repository.")
    except Exception as e:
        print(f"Errore generico durante l'inserimento del repository {nameRep}: {e}")

print("Tutte le informazioni delle PR sono state salvate in MongoDB.")
