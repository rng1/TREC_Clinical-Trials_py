import os

import pandas as pd
import torch
from torch import nn

TRIAL_CSV_PATH = "C:/Users/rnara/Desktop/TFG/[py] TREC_Clinical-Trials/trial_embeddings.csv"
TOPIC_CSV_PATH = "C:/Users/rnara/Desktop/TFG/[py] TREC_Clinical-Trials/topic_embeddings.csv"


def get_embeddings(tokenizer, model: nn.Module, text: str) -> torch.Tensor:
    model = model.cuda()
    outputs = []
    device = next(model.parameters()).device
    max_len = int(model.config.max_position_embeddings)
    overlap = int(max_len * 0.02)
    window_size = max_len - overlap

    inputs = tokenizer(text, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False)

    if inputs.input_ids.shape[1] <= overlap:
        out = model(**{name: x.to(device) for name, x in inputs.items()})
        outputs.append(out.last_hidden_state[:, 0, :])
    else:
        for i in range(overlap, inputs.input_ids.shape[1], window_size):
            start, end = (i - overlap), (i + window_size)
            out = model(**{name: x[:, start:end].to(device) for name, x in inputs.items()})
            outputs.append(out.last_hidden_state[:, 0, :])

    return torch.stack(outputs, 1).mean(1).to("cpu")


def get_trial_embeddings(tokenizer, model: nn.Module, trial_csv: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(TRIAL_CSV_PATH):
        print("Loading trial embeddings from CSV...")
        return pd.read_csv(TRIAL_CSV_PATH, sep=";")
    else:
        print("Generating trial embeddings...")
        return generate_trial_embeddings(tokenizer, model, trial_csv)


def get_topic_embeddings(tokenizer, model: nn.Module, topic_csv: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(TOPIC_CSV_PATH):
        print("Loading topic embeddings from CSV...")
        return pd.read_csv(TOPIC_CSV_PATH, sep=";")
    else:
        print("Generating topic embeddings...")
        return generate_topic_embeddings(tokenizer, model, topic_csv)


def generate_trial_embeddings(tokenizer, model: nn.Module, trial_csv: pd.DataFrame) -> pd.DataFrame:
    data = []
    embeddings = []
    for _, trial in trial_csv.iterrows():
        print(f"Processing trial {trial['nct_id']}...")
        summary = trial["summary"] if pd.notna(trial["summary"]) else ""
        description = trial["description"] if pd.notna(trial["description"]) else ""
        criteria = trial["criteria"] if pd.notna(trial["criteria"]) else ""
        text = summary + description + criteria
        embeddings = get_embeddings(tokenizer, model, text).detach().numpy().tolist()[0]
        data_row = [trial["nct_id"]] + embeddings
        data.append(data_row)

    columns = ["nct_id"] + [f"embedding_{i + 1}" for i in range(len(embeddings))]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(TRIAL_CSV_PATH, sep=";", index=False)

    return df


def generate_topic_embeddings(tokenizer, model: nn.Module, topic_csv: pd.DataFrame) -> pd.DataFrame:
    data = []
    embeddings = []
    for _, topic in topic_csv.iterrows():
        print(f"Processing topic {topic['id']}...")
        embeddings = get_embeddings(tokenizer, model, topic["description"]).detach().numpy().tolist()[0]
        data_row = [topic["id"]] + embeddings
        data.append(data_row)

    columns = ["id"] + [f"embedding_{i + 1}" for i in range(len(embeddings))]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(TOPIC_CSV_PATH, sep=";", index=False)

    return df
