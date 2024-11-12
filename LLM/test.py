import requests
import json
import pymongo
import random
from bson import json_util

# Connessione al database MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["github"]
collection = db["pull_requests_new"]

# Estrai una pull request casuale dalla collection e rimuovi il campo "body_message"
pull_request = collection.aggregate([{"$sample": {"size": 1}}]).next()

# Stampa la pull request recuperata per vedere i dati che vengono esaminati
print("Pull Request Recuperata:", json.dumps(pull_request, indent=4, default=json_util.default))

actual_body_message = pull_request.pop("body_message", None)  # Rimuove il campo 'body_message' dalla pull request

# Ottieni eventuale issue associata alla pull request
issue = pull_request.get("issue", "N/A")

# Definisci l'URL del server Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# Testo di input da fornire al modello
prompt = f"""
As an AI specializing in generating pull request messages for Java-based repositories, provide pull request message predictions in a structured format.

Your tasks are to:
1. Generate predictions exclusively in the form of pull request messages that align with the content and context of the pull request.
2. Ensure that predictions are contextually accurate based on provided repository data.
3. Focus solely on suggesting meaningful pull request messages that clearly convey the purpose of the changes.
4. Avoid generic pull request messages like "minor changes" or "updated code".
5. Consider the content of the commit messages, titles, diff, and any associated issue to generate appropriate pull request messages.
6. Ignore any code details or syntax; focus solely on the purpose and description of the pull request.
7. your response must be in the format "body_message: *predicted response* "
Data for predicted pull request message:
Commit Messages: {pull_request.get('commit_message')}
Title: {pull_request.get('title')}
Diff: {pull_request.get('diff')}
Issue: {issue}
"""

# Payload per la richiesta
payload = {
    "model": "llama3.2:latest",  # Usa il nome del modello che hai installato
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

    except json.JSONDecodeError as e:
        # Stampa il contenuto della risposta in caso di errore
        print("Errore nel decodificare la risposta JSON:", e)
        print("Contenuto della risposta:")
        print(response.text)
else:
    print(f"Errore: {response.status_code}, {response.text}")

