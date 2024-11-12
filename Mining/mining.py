import json
import os.path
from github import Github, Auth

auth = Auth.Token("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")
g = Github(auth=auth)

output_file = "repositories_java.json"

# Funzione per caricare i repository esistenti
def load_existing_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

# Verifica se un repository è già presente nel file JSON
def is_duplicate(repo_id, existing_ids):
    return repo_id in existing_ids

# Salvataggio dati esistenti
old_data = load_existing_data(output_file)
existing_ids = set(repo['id'] for repo in old_data)

# Lista per memorizzare i nuovi dati
repos_data = []

# Funzione per eseguire query con limiti di stelle
def search_repos_with_star_range(query, min_stars, max_stars):
    try:
        star_query = f"{query} stars:{min_stars}..{max_stars}"
        repositories = g.search_repositories(star_query)
        return repositories
    except Exception as e:
        print(f"Errore durante la ricerca: {e}")
        return None

query_base = "forks:>50 language:java"
per_page = 100

# Intervalli di stelle da usare per le query
star_ranges = [
    (500, 600),
    (601, 700),
    (701, 800),
    (801, 900),
    (901, 1000),
    (1001, 1400),
    (1401, 2000),
    (2001, 4000),
    (4001, 200000),
]

# Itera su ciascun intervallo di stelle
for min_stars, max_stars in star_ranges:
    repositories = search_repos_with_star_range(query_base, min_stars, max_stars)

    if repositories is None:
        break

    total_repos = repositories.totalCount
    total_pages = (total_repos // per_page) + 1  # Calcola il numero di pagine

    print(f"Scaricando repository con stelle tra {min_stars} e {max_stars}: {total_repos} trovati")

    # Ciclo per paginazione
    for page in range(total_pages):
        try:
            repos_page = repositories.get_page(page)
            for repo in repos_page:
                repo_info = repo.raw_data
                if not is_duplicate(repo_info['id'], existing_ids):
                    repos_data.append(repo_info)
        except Exception as e:
            print(f"Errore durante il recupero della pagina {page}: {e}")
            continue

# Unione dei 'vecchi' e 'nuovi' dati
old_data.extend(repos_data)

# Salvataggio repository in JSON
with open(output_file, 'w') as f:
    json.dump(old_data, f, indent=4)

print(f"Numero di nuovi repository aggiunti: {len(repos_data)}")
print(f"Dati salvati con successo in {output_file}")

