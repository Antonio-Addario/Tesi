from pymongo.mongo_client import MongoClient

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']
projects = db['projects']
pull_requests_collection = db['pull_requests_new']

# Trova tutti i repository nella collezione 'projects'
repositories = projects.find()

# Dati da salvare
repo_pr_counts = []

print("Conteggio delle pull request per ogni repository:")
for repo in repositories:
    repository_id = repo["_id"]
    repository_name = repo["repository_name"]

    # Conta il numero di pull request associate al repository
    pr_count = pull_requests_collection.count_documents({"repository_id": repository_id})

    # Stampa il risultato
    print(f"Repository: {repository_name}, Pull Requests: {pr_count}")

    # Aggiungi ai dati da salvare
    repo_pr_counts.append({
        "repository_name": repository_name,
        "pull_request_count": pr_count
    })

# Salva i risultati in una collezione separata (opzionale)
repo_counts_collection = db['repo_pr_counts']
repo_counts_collection.delete_many({})  # Pulisce i dati esistenti
repo_counts_collection.insert_many(repo_pr_counts)
print("I conteggi delle pull request sono stati salvati nella collezione 'repo_pr_counts'.")
