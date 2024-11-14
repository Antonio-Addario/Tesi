from pymongo import MongoClient
import pickle
from sklearn.model_selection import train_test_split

def split_dataset_by_repository(repository_name, test_size=0.2, random_state=42):

    client = MongoClient("mongodb://localhost:27017/")
    db = client['github']
    projects = db['projects']
    pull_requests = db['pull_requests_new']

    repository_id = projects.find_one({"repository_name": repository_name})['_id']
    pull_requests = list(pull_requests.find({"repository_id": repository_id}))

    if not pull_requests:
        print(f"Nessuna pull request trovata per il repository: {repository_name}")
        return

    print(f"Trovate {len(pull_requests)} pull request per il repository: {repository_name}")

    train_set, test_set = train_test_split(pull_requests, test_size=test_size, random_state=random_state)

    print(f"Training set: {len(train_set)} PRs, Test set: {len(test_set)} PRs")

    # Salva i risultati su file
    with open("train_set.pkl", "wb") as train_file:
        pickle.dump(train_set, train_file)

    with open("test_set.pkl", "wb") as test_file:
        pickle.dump(test_set, test_file)

    print("Dataset separato con successo e salvato nei file 'train_set.pkl' e 'test_set.pkl'.")

split_dataset_by_repository(
    repository_name="bitcoinj/bitcoinj"
)
