import pickle

import pandas as pd


class FilterTrials:
    CACHE_PATH = "filtered_trials_cache.pkl"

    def __init__(self, trials: pd.DataFrame, topics: pd.DataFrame):
        self.filtered_dict = self.load_cache()

        if not self.filtered_dict:
            self.filter_trials(trials, topics)
            self.save_cache()

    def filter_trials(self, trials: pd.DataFrame, topics: pd.DataFrame):
        for _, topic in topics.iterrows():
            filtered_trials = []
            for _, trial in trials.iterrows():
                if ((trial.gender == topic.gender or trial.gender == "all")
                        and trial.min_age <= topic.age <= trial.max_age):
                    filtered_trials.append(trial.nct_id)
            self.filtered_dict[topic.id] = filtered_trials

    def load_cache(self):
        try:
            with open(self.CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_cache(self):
        with open(self.CACHE_PATH, "wb") as f:
            pickle.dump(self.filtered_dict, f)
