from pymongo import MongoClient

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']
pull_requests_collection = db['pull_requests']

# Nome del repository da filtrare
repository_name = "bitcoinj/bitcoinj"

# Recupera il documento del repository specifico
repository_document = pull_requests_collection.find_one({"repository_name": repository_name})

if repository_document:
    total_pr_count = len(repository_document.get("pull_requests", []))
    print(f"Il numero totale di pull request salvate per il repository '{repository_name}' è: {total_pr_count}")
else:
    print(f"Nessun documento trovato per il repository '{repository_name}'")
