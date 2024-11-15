import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import requests
import json
from nltk.translate.bleu_score import sentence_bleu


def load_datasets():
    with open("train_set.pkl", "rb") as train_file:
        train_set = pickle.load(train_file)

    with open("test_set.pkl", "rb") as test_file:
        test_set = pickle.load(test_file)

    # Limita il training set a 20 documenti
    train_set = train_set[:20]

    print(f"Training set caricato con {len(train_set)} PRs.")
    print(f"Test set caricato con {len(test_set)} PRs.")
    return train_set, test_set



def create_faiss_index(train_set):
    pr_texts = [
        f"Title: {pr['title']} "
        f"Commit Message: {pr['commit_message']} "
        f"Issue: {pr['issue']['title'] if pr.get('issue') else 'Nessuna issue associata'} "
        f"Comments: {' '.join(pr['issue'].get('comments', [])) if pr.get('issue') and pr['issue'].get('comments') else 'Nessun commento'}"
        for pr in train_set
        if pr.get('title') and pr.get('commit_message')
    ]


    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(pr_texts)
    print("qui ci sono")

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
    try:
        print("Caricamento del modello...")
        # Crash -> Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")

    query_embedding = model.encode([query])
    print("non sono qui")
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Cerca i documenti simili nell'indice FAISS
    distances, indices = index.search(query_embedding, top_k)

    return [train_texts[idx] for idx in indices[0]]


def generate_body_message(test_pr, retrieved_context):

    # Estrai informazioni dall'issue associata
    issue_info = test_pr.get('issue', {})
    issue_title = issue_info.get('title', 'Nessuna issue associata')
    issue_comments = " ".join(issue_info.get('comments', []))

    prompt = f"""
    Sei un assistente esperto nello sviluppo software.
    Devi completare il campo body_message di una pull request dato il suo contesto.

    Ecco la pull request da completare:
    Title: {test_pr['title']}
    Commit Message: {test_pr['commit_message']}
    Diff: {test_pr['diff']}
    Issue associata: {issue_title}
    Commenti sull'issue: {issue_comments}

    Qui sotto trovi alcune pull request simili per contesto:
    {retrieved_context}

    Completa il campo body_message per questa pull request.
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


def evaluate(test_set):
    scores = []
    for i, test_pr in enumerate(test_set):
        if not isinstance(test_pr, dict):
            print(f"Errore: test_pr non è un dizionario. Indice: {i}, Tipo: {type(test_pr)}, Contenuto: {test_pr}")
            continue

        # Verifica che i campi richiesti siano presenti
        title = test_pr.get('title', 'Titolo non disponibile')
        commit_message = test_pr.get('commit_message', 'Messaggio non disponibile')
        body_message = test_pr.get('body_message', '')
        issue_title = test_pr.get('issue', {}).get('title', 'Nessuna issue associata')
        if not title or not commit_message:
            print(f"Errore: PR mancante di dati essenziali. Indice: {i}, Contenuto: {test_pr}")
            continue

        # Prepara la query per il recupero del contesto
        query = f"Title: {title} Commit Message: {commit_message} Body Message: {body_message} Issue: {issue_title}"

        # Recupera il contesto dal training set
        context = retrieve_context(query, top_k=3)

        # Rimuovi il body_message dalla PR prima di passarla al modello
        pr_without_body = test_pr.copy()
        pr_without_body.pop('body_message', None)

        # Genera il body_message
        predicted_body = generate_body_message(pr_without_body, context)

        # Calcola il BLEU score
        score = sentence_bleu([body_message.split()], predicted_body.split())
        scores.append(score)

        print(f"Real Body: {body_message}")
        print(f"Predicted Body: {predicted_body}")
        print(f"BLEU Score: {score}")

    return sum(scores) / len(scores) if scores else 0.0




if __name__ == "__main__":
    train_set, test_set = load_datasets()

    # Crea l'indice FAISS con il training set
    create_faiss_index(train_set)

    # Valuta le prestazioni sul test set
    average_bleu = evaluate(test_set)
    print(f"BLEU Score Medio: {average_bleu}")
