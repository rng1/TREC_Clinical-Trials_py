from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from numpy import linalg as LA

import torch

from embeddings import get_trial_embeddings, get_topic_embeddings
from filter import FilterTrials
from xmlparser import TopicParser
from xmlparser import TrialParser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device used : {device}")

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base").to(device).requires_grad_(False)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    topic_csv = TopicParser().df
    topic_embeddings = get_topic_embeddings(tokenizer=tokenizer, model=model, topic_csv=topic_csv)
    trial_csv = TrialParser().df
    filtered_dict = FilterTrials(trials=trial_csv, topics=topic_csv).filtered_dict
    unique_trials = list(set(sum(filtered_dict.values(), [])))
    trial_csv = trial_csv[trial_csv["nct_id"].isin(unique_trials)]
    trial_embeddings = get_trial_embeddings(tokenizer=tokenizer, model=model, trial_csv=trial_csv)

    topics = topic_embeddings.iloc[:, 1:].values
    trials = trial_embeddings.iloc[:, 1:].values
    topics_norm = LA.norm(topics, axis=1).reshape(-1, 1)
    trials_norm = LA.norm(trials, axis=1).reshape(1, -1)
    topic_trial_dot_product = topics@trials.T
    cosine_similarity = topic_trial_dot_product / (topics_norm * trials_norm)

