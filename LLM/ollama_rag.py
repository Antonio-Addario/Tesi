
import requests
import json
import config as cf
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle


# corpus_of_documents = [
#     "Take a leisurely walk in the park and enjoy the fresh air.",
#     "Visit a local museum and discover something new.",
#     "Attend a live music concert and feel the rhythm.",
#     "Go for a hike and admire the natural scenery.",
#     "Have a picnic with friends and share some laughs.",
#     "Explore a new cuisine by dining at an ethnic restaurant.",
#     "Take a yoga class and stretch your body and mind.",
#     "Join a local sports league and enjoy some friendly competition.",
#     "Attend a workshop or lecture on a topic you're interested in.",
#     "Visit an amusement park and ride the roller coasters."
# ]


def load_and_index_documents_from_files(root_kb):
    """
    This function takes a directory path, reads the content of each file,
    converts them into embeddings, and indexes them using FAISS. It also saves
    the embeddings and the FAISS index to the local disk for future use.

    Parameters:
    root_kb (str): Path to the directory containing the document files.

    Returns:
    index (faiss.IndexFlatL2): The FAISS index containing the document embeddings.
    document_texts (list of str): List of document texts that were indexed.
    """

    document_texts = []

    # Read the content of each file
    for file_name in os.listdir(root_kb):
        file_path = os.path.join(root_kb, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
            document_texts.append(document_text)

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert the documents into embeddings
    embeddings = model.encode(document_texts)

    # Normalize the embeddings to unit vectors
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Save embeddings and document_texts to disk
    np.save('embeddings.npy', embeddings)
    with open('document_texts.pkl', 'wb') as f:
        pickle.dump(document_texts, f)

    # Create a FAISS index (IndexFlatL2: simple, flat index for L2 distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add the embeddings to the FAISS index
    index.add(embeddings)

    # Save the FAISS index to disk
    faiss.write_index(index, 'faiss.index')

    return


def retrieve_documents(query, top_k=3):
    """
    Retrieves the top-k relevant documents based on the query by loading the
    precomputed FAISS index and document embeddings from the local disk.

    Parameters:
    query (str): The search query.
    top_k (int): The number of top results to return.

    Returns:
    top_docs (list of str): List of the top-k most relevant documents.
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    import pickle

    # Load the saved FAISS index
    index = faiss.read_index('faiss.index')

    # Load the saved document_texts
    with open('document_texts.pkl', 'rb') as f:
        document_texts = pickle.load(f)

    # Load the same sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the query to the same vector space as the documents
    query_embedding = model.encode([query])

    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Search the FAISS index for top_k nearest neighbors
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the corresponding documents
    top_docs = [document_texts[idx] for idx in indices[0]]

    return top_docs

# def jaccard_similarity(query, document):
#     query = query.lower().split(" ")
#     document = document.lower().split(" ")
#     intersection = set(query).intersection(set(document))
#     union = set(query).union(set(document))
#     return len(intersection)/len(union)

# def return_response(query, corpus):
#     similarities = []
#     for doc in corpus:
#         similarity = jaccard_similarity(query, doc)
#         similarities.append(similarity)
#     return corpus_of_documents[similarities.index(max(similarities))]


# https://github.com/jmorganca/ollama/blob/main/docs/api.md

def generate_answer():
    user_input = ("Retrieve the number of forks and stars of this repository: https://github.com/PyGithub/PyGithub"
            )


    #index, document_texts = load_and_index_documents_from_files(cf.KB_SRC)
    relevant_document = retrieve_documents(user_input)

    prompt = f"""
    You are a mining assistant. Help the user in mining open source repositories.
    The user input is: {user_input}   
    The PygitHub documentation is available here {relevant_document}
    Generate the CSV containing the requested information.
    """
    full_response = []
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llama3.1",
       "prompt": prompt.format(user_input=user_input, relevant_document=relevant_document),
        "max_tokens": 100,
        "temperature": 0.3

    }


    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)
    try:
        count = 0
        for line in response.iter_lines():
            # filter out keep-alive new lines
            # count += 1
            # if count % 5== 0:
            #     print(decoded_line['response']) # print every fifth token
            if line:
                decoded_line = json.loads(line.decode('utf-8'))

                full_response.append(decoded_line['response'])
    finally:
        response.close()
    print(''.join(full_response))







#preprocess_ows_file()
generate_answer()