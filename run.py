from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from numpy import linalg as LA

import torch
import numpy as np

from embeddings import get_trial_embeddings, get_topic_embeddings
from filter import FilterTrials
from xmlparser import TopicParser
from xmlparser import TrialParser


def map_index_to_id(index):
    return trial_ids[index]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device used : {device}")

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base").to(device).requires_grad_(False)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

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
    capped_indices = sorted_indices[:, :1000]

    vfunc = np.vectorize(map_index_to_id)

    unfiltered_ranking = vfunc(capped_indices)

    # Filtered ranking
    sorted_capped_idx = np.zeros((50, 1000), dtype=int)
    threshold = 1000
    trial_id_to_index = {id: index for index, id in enumerate(trial_csv['nct_id'])}
    filtered_idx = {topic: [trial_id_to_index[id] for id in ids] for topic, ids in filtered_dict.items()}

    for i in range(sorted_indices.shape[0]):
        mask = np.isin(sorted_indices[i], filtered_idx[i+1])
        filtered_row = np.pad(sorted_indices[i][mask], (0, threshold), mode='constant', constant_values=-1)[:threshold]
        sorted_capped_idx[i] = filtered_row

    filtered_ranking = vfunc(sorted_capped_idx)
