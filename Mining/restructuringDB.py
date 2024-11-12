from pymongo.mongo_client import MongoClient

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['github']

# Collection esistente
old_collection = db['pull_requests']

# Nuove collection
projects_collection = db['projects']
pull_requests_collection = db['pull_requests_new']

# Itera su tutti i documenti della collection esistente
for old_doc in old_collection.find():
    project_name = old_doc["repository_name"]

    # Inserisci il progetto nella collection `projects` se non esiste già
    if not projects_collection.find_one({"repository_name": project_name}):
        project_data = {
            "repository_name": project_name,
        }
        projects_collection.insert_one(project_data)
        print(f"Inserito progetto: {project_name}")

    # Itera su ogni pull request del documento del progetto
    for pr in old_doc["pull_requests"]:
        pull_request_data = {
            "repository_id": projects_collection.find_one({"repository_name": project_name}).get("_id"),  # Riferimento al progetto
            "title": pr["title"],
            "body_message": pr.get("body_message", ""),
            "commit_message": pr.get("commit_message", "N/A"),
            "diff": pr.get("diff", ""),
            "issue": pr.get("issue"),
            "created_at": pr.get("created_at")
        }

        # Inserisci la pull request nella nuova collection `pull_requests`
        pull_requests_collection.insert_one(pull_request_data)
        print(f"Inserita PR per il progetto {project_name}: {pr['title']}")

print("Ristrutturazione completata.")
