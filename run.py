# from adapters import AutoAdapterModel

import numpy as np
import torch
from numpy import linalg as LA
from transformers import AutoTokenizer, AutoModel

from embeddings import get_trial_embeddings, get_topic_embeddings
from filter import FilterTrials
from xmlparser import TopicParser
from xmlparser import TrialParser


SPECTER = "allenai/specter2_base"
BIOBERT = "dmis-lab/biobert-v1.1"
CLINICALBERT = "medicalai/ClinicalBERT"
BIOCLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"

MODEL = SPECTER


def create_trec_eval_file(ranking_scores, ranking_ids, run_name):
    with open(f'py_{run_name.lower()}.txt', 'w') as f:
        for i in range(ranking_scores.shape[0]):
            for j in range(ranking_scores.shape[1]):
                line = f"{i + 1} Q0 {ranking_ids[i][j].upper()} {j + 1} {ranking_scores[i][j]} {run_name.upper()}\n"
                f.write(line)


def map_index_to_id(index):
    return trial_ids[index]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device used : {device}")
print(f"Model used : {MODEL}\n")

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).requires_grad_(False)

    topic_csv = TopicParser().df
    trial_csv = TrialParser().df
    filtered_dict = FilterTrials(trials=trial_csv, topics=topic_csv).filtered_dict
    unique_trials = list(set(sum(filtered_dict.values(), [])))
    trial_csv = trial_csv[trial_csv["nct_id"].isin(unique_trials)]
    trial_ids = trial_csv['nct_id'].tolist()
    topic_embeddings = get_topic_embeddings(tokenizer=tokenizer, model=model, topic_csv=topic_csv)
    trial_embeddings = get_trial_embeddings(tokenizer=tokenizer, model=model, trial_csv=trial_csv)

    topics = topic_embeddings.iloc[:, 1:].values
    trials = trial_embeddings.iloc[:, 1:].values
    topics_norm = LA.norm(topics, axis=1).reshape(-1, 1)
    trials_norm = LA.norm(trials, axis=1).reshape(1, -1)
    topic_trial_dot_product = topics @ trials.T
    cosine_similarity = topic_trial_dot_product / (topics_norm * trials_norm)
    sorted_indices = np.argsort(cosine_similarity, axis=1)[:, ::-1]
    threshold = 1000
    capped_indices = sorted_indices[:, :threshold]

    vfunc = np.vectorize(map_index_to_id)

    unfiltered_ranking = vfunc(capped_indices)

    # Filtered ranking
    sorted_capped_indices = np.zeros((50, threshold), dtype=int)
    trial_id_to_index = {id: index for index, id in enumerate(trial_csv['nct_id'])}
    filtered_idx = {topic: [trial_id_to_index[id] for id in ids] for topic, ids in filtered_dict.items()}

    for i in range(sorted_indices.shape[0]):
        mask = np.isin(sorted_indices[i], filtered_idx[i+1])
        filtered_row = np.pad(sorted_indices[i][mask], (0, threshold), mode='constant', constant_values=-1)[:threshold]
        sorted_capped_indices[i] = filtered_row

    filtered_ranking = vfunc(sorted_capped_indices)

    # Get the ranking scores
    row_indices = np.arange(cosine_similarity.shape[0])[:, None]
    unfiltered_ranking_scores = cosine_similarity[row_indices, capped_indices]
    filtered_ranking_scores = cosine_similarity[row_indices, sorted_capped_indices]

    # Create the trec_eval files
    create_trec_eval_file(unfiltered_ranking_scores, unfiltered_ranking, "rocchio_test")
    create_trec_eval_file(filtered_ranking_scores, filtered_ranking, "filtered")
