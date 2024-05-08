import re


class Topic:
    def __init__(self, topic_id: int, description: str):
        self.topic_id = topic_id
        self.age = self.parse_age(description)
        self.gender = self.parse_gender(description)

        description = description.replace("\n", "").replace("\r", "")
        self.description = re.sub(" +", " ", description)

    @staticmethod
    def parse_gender(description: str) -> str:
        male_identifiers = ["male", "man", "boy", "he", "his"]
        female_identifiers = ["female", "woman", "girl", "she", "her"]

        description_words = description.split()

        for word in description_words:
            if word in male_identifiers:
                return "male"
            elif word in female_identifiers:
                return "female"

        return ""

    @staticmethod
    def parse_age(description: str) -> float:
        age_pattern = re.compile(r"(\d+)-(year|month|week)", re.IGNORECASE)
        match = age_pattern.search(description)
        if match:
            digit = int(match.group(1))
            unit = match.group(2)

            if unit == "year":
                return digit
            elif unit == "month":
                return digit / 12
            elif unit == "week":
                return digit / 52
        return -1
