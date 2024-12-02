import os
import statistics
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import requests
import json
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import nltk
from bson import ObjectId

nltk.download('wordnet')
import re

model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_diff_details(diff_text):
    comments = re.findall(r'//.*|/\*.*?\*/', diff_text, re.DOTALL)
    class_names = re.findall(r'\bclass\s+(\w+)', diff_text)
    method_names = re.findall(r'\b(?:public|private|protected)?\s+[\w<>\[\]]+\s+(\w+)\s*\(.*?\)\s*{', diff_text)
    pom_add_dependencies = re.findall(r'<dependency>.*?<artifactId>(.*?)</artifactId>.*?</dependency>', diff_text,
                                      re.DOTALL)
    pom_remove_dependencies = re.findall(r'<!--.*?REMOVE.*?<artifactId>(.*?)</artifactId>.*?-->', diff_text, re.DOTALL)

    return {
        "comments": comments,
        "class_names": class_names,
        "method_names": method_names,
        "added_dependencies": pom_add_dependencies,
        "removed_dependencies": pom_remove_dependencies
    }


def create_faiss_index(train_set):
    pr_texts = [
        f"title: {pr['title']} "
        f"body_message: {pr['body_message']} "
        f"commit_message: {pr['commit_message']} "
        f"diff: {extract_diff_details(pr['diff'])} "
        f"issue: {pr['issue']['title'] if pr.get('issue') else None} "
        for pr in train_set
        if pr.get('title') and pr.get('commit_message') and len(pr.get("body_message", "")) > 40
    ]
    embeddings = model.encode(pr_texts)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save('embeddings.npy', embeddings)
    with open('train_texts.pkl', 'wb') as f:
        pickle.dump(pr_texts, f)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, 'faiss.index')

    print("Indice FAISS creato e salvato.")


def retrieve_context(query, top_k=3):
    # Carica l'indice FAISS e i testi del training set
    index = faiss.read_index('faiss.index')

    with open('train_texts.pkl', 'rb') as f:
        train_texts = pickle.load(f)
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Cerca i documenti simili nell'indice FAISS
    distances, indices = index.search(query_embedding, top_k)

    return [train_texts[idx] for idx in indices[0]]


def generate_body_message(test_pr, retrieved_context, train_set):
    # Estrai informazioni dall'issue associata
    issue_info = test_pr.get('issue', {})
    diff = extract_diff_details(test_pr["diff"])
    body_messages = [pr["body_message"] for pr in train_set if "body_message" in pr and len(pr["body_message"]) > 40]
    # Lista delle lunghezze di tutti i "body_message"
    len_body_message = [len(msg) for msg in body_messages]
    if len(len_body_message) > 0:  # Assicurati che ci siano dati
        media = statistics.mean(len_body_message)
        deviazione = statistics.stdev(len_body_message)
        min_len = min(len_body_message)
        max_len = max(len_body_message)

        print(f"Media delle lunghezze: {media}")
        print(f"Deviazione standard: {deviazione}")
        print(f"Lunghezza minima: {min_len}")
        print(f"Lunghezza massima: {max_len}")
    else:
        print("Nessun dato disponibile per calcolare statistiche.")

    prompt = f"""
    Generate a pull request body message that aligns with the provided context.
    The length should be similar to previous messages:
    - Average: {media:.2f} characters, Min: {min_len}, Max: {max_len}, StdDev: {deviazione:.2f}

    Information about the current pull request:
    - title: {test_pr['title']}
    - commit_message: {test_pr['commit_message']}
    - diff: {diff}
    - issue: {issue_info if issue_info else 'None'}

    Similar pull requests for context:
    {retrieved_context}

    Please generate the body message in the format:
    body_message: *your response here*
    """

    # Chiamata all'API del modello LLM
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.3
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

    full_response = []
    for line in response.iter_lines():
        if line:
            decoded_line = json.loads(line.decode('utf-8'))
            full_response.append(decoded_line['response'])

    return ''.join(full_response)


def evaluate(test_set, train_set):
    bleu_scores = []
    meteor_scores = []
    pull_requests_test = [
        {**pr, 'diff': extract_diff_details(pr.get('diff', ""))}
        for pr in test_set
        if len(pr.get("body_message", "")) > 40 and pr.get('title') and pr.get('commit_message')
    ]

    for test_pr in pull_requests_test:
        diff = extract_diff_details(test_pr.get("diff"))
        # Prepara la query per il recupero del contesto
        query = (f"title: {test_pr.get('title')} "
                 f"body_message: {test_pr.get('body_message')} "
                 f"commit_message: {test_pr.get('commit_message')} "
                 f"diff: {diff} "
                 f"issue: {test_pr.get('issue')}")

        # Recupera il contesto dal training set
        context = retrieve_context(query, top_k=3)

        # Rimuovi il body_message dalla PR prima di passarla al modello
        pr_without_body = test_pr.copy()
        body_message = test_pr.get("body_message")
        pr_without_body.pop('body_message', None)

        # Genera il body_message
        predicted_body = generate_body_message(pr_without_body, context, train_set)

        # Calcola il BLEU score
        bleu_score = sentence_bleu([body_message.split()], predicted_body.split())
        meteor_score = single_meteor_score(body_message, predicted_body)

        bleu_scores.append(bleu_score)
        meteor_scores.append(meteor_score)

        print(f"Real Body: {body_message}")
        print(f"Predicted Body: {predicted_body}")
        print(f"BLEU Score: {bleu_score}")
        print(f"METEOR Score: {meteor_score}")

    # Calcola i punteggi medi
    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    average_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return average_bleu, average_meteor


def save_to_json(data, filename):

    def convert(obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=convert)


def load_from_json(filename):

    with open(filename, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # client = MongoClient("mongodb://localhost:27017/")
    # db = client['github']
    # projects = db['projects']
    # pull_requests = db['pull_requests_new']
    # repository_name = "DiUS/java-faker"
    # repository_id = projects.find_one({"repository_name": repository_name})['_id']
    # pull_requests = list(pull_requests.find({"repository_id": repository_id}))
    #
    # if not pull_requests:
    #     print(f"Nessuna pull request trovata per il repository: {repository_name}")
    #     exit()

    train_file = 'train_set.json'
    test_file = 'test_set.json'

    # Carica o crea train/test set
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Caricamento del train e test set dai file...")
        train_set = load_from_json(train_file)
        test_set = load_from_json(test_file)
    # else:
    #     print("Creazione del train e test set...")
    #     train_set, test_set = train_test_split(pull_requests, test_size=0.2, random_state=42)
    #     save_to_json(train_set, train_file)
    #     save_to_json(test_set, test_file)

    print(f"Training set: {len(train_set)} PRs, Test set: {len(test_set)} PRs")
    # Crea l'indice FAISS con il training set
    create_faiss_index(train_set)

    # Valuta le prestazioni sul test set
    average_bleu, average_meteor = evaluate(test_set, train_set)
    print(f"BLEU Score Medio: {average_bleu}")
    print(f"METEOR Score Medio: {average_meteor}")
