import json
import pymongo
import random
import statistics
import re
import requests
import nltk
from bson import ObjectId
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score

nltk.download('wordnet')

def extract_diff_details(diff_text):
    comments = re.findall(r'//.*|/\*.*?\*/', diff_text, re.DOTALL)
    class_names = re.findall(r'\bclass\s+(\w+)', diff_text)
    method_names = re.findall(r'\b(?:public|private|protected)?\s+[\w<>\[\]]+\s+(\w+)\s*\(.*?\)\s*{', diff_text)
    pom_add_dependencies = re.findall(r'<dependency>.*?<artifactId>(.*?)</artifactId>.*?</dependency>', diff_text, re.DOTALL)
    pom_remove_dependencies = re.findall(r'<!--.*?REMOVE.*?<artifactId>(.*?)</artifactId>.*?-->', diff_text, re.DOTALL)

    return {
        "comments": comments,
        "class_names": class_names,
        "method_names": method_names,
        "added_dependencies": pom_add_dependencies,
        "removed_dependencies": pom_remove_dependencies
    }

# Connessione al database MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["github"]
pull_requests_collection = db["pull_requests_new"]

# ID del repository da cercare
repository_id = ObjectId("672a3cfd8eb967273d92b957")

pull_requests = list(pull_requests_collection.find({
    "repository_id": repository_id,
    "body_message": {
        "$exists": True,
        "$ne": None
    }
}))
pull_requests = [
    {**pr, 'diff': extract_diff_details(pr.get('diff', ""))}
    for pr in pull_requests
    if len(pr.get("body_message", "")) > 40
]

# Selezionare una pull request a caso da prevedere
if pull_requests:
    actual_pr = random.choice(pull_requests)
    pull_requests.remove(actual_pr)
    actual_body_message = actual_pr.pop("body_message", None)
    print("Body_message da prevedere: ", actual_body_message)
    diff_details = actual_pr.get('diff', "")
    formatted_diff_details = (
        f"Comments: {diff_details['comments']}\n"
        f"Class Names: {diff_details['class_names']}\n"
        f"Method Names: {diff_details['method_names']}\n"
        f"Added Dependencies: {diff_details['added_dependencies']}\n"
        f"Removed Dependencies: {diff_details['removed_dependencies']}\n"
    )

    # Assicurati che esistano dati validi e che il campo "body_message" sia presente
    body_messages = [pr["body_message"] for pr in pull_requests if "body_message" in pr and len(pr["body_message"]) > 40]

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

    # Limita il contesto a 3 pull request simili
    similar_pull_requests = pull_requests[:3]
    context_pull_requests = [{
        'title': pr.get('title'),
        'body_message': pr.get('body_message'),
        'commit_message': pr.get('commit_message')
    } for pr in similar_pull_requests]

# Definisci l'URL del server Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# Testo di input da fornire al modello
prompt = f"""
    Based on the provided information, generate a pull request body message that:
    - Avoids unnecessary descriptions and focuses only on the body message.
    - The length of the message should be approximately similar to the average length of previous messages.
    Length Statistics for Reference:
    - Average Length: {media:.2f} characters
    - Minimum Length: {min_len} characters
    - Maximum Length: {max_len} characters
    - Standard Deviation: {deviazione:.2f} characters

    Data for the pull request:
    Commit Messages: {actual_pr.get('commit_message')}
    Title: {actual_pr.get('title')}
    Issue: {actual_pr.get('issue')}
    """

# Payload per la richiesta
payload = {
    "model": "llama3.2:latest",
    "prompt": prompt
}

# Effettua la richiesta al server Ollama
response = requests.post(OLLAMA_URL, data=json.dumps(payload))

# Verifica lo stato della richiesta e stampa il contenuto della risposta
if response.status_code == 200:
    try:
        # Split the response by lines and parse each line as JSON
        output = ""
        for line in response.text.splitlines():
            # Convert the line to a dictionary and concatenate the response part
            json_line = json.loads(line)
            output += json_line["response"]

        # Stampa l'output completo
        print("Predicted BODY MESSAGE:", output)
        print("Actual BODY MESSAGE:", actual_body_message)
        bleu_smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([actual_body_message.split()], output.split(), smoothing_function=bleu_smoothing)
        print("BLEU SCORE:", bleu_score)
        meteor_score = single_meteor_score(actual_body_message.split(), output.split())
        print("METEOR SCORE:", meteor_score)

    except json.JSONDecodeError as e:
        # Stampa il contenuto della risposta in caso di errore
        print("Errore nel decodificare la risposta JSON:", e)
        print("Contenuto della risposta:")
        print(response.text)
else:
    print(f"Errore: {response.status_code}, {response.text}")
