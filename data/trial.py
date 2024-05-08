from typing import List


class Trial:
    def __init__(
            self,
            nct_id: str,
            gender: str,
            min_age: str,
            max_age: str,
            keywords: List[str],
            conditions: List[str],
            title: str,
            summary: str,
            description: str,
            criteria: str
    ):
        self.nct_id = nct_id
        self.gender = self.parse_gender(gender)
        self.min_age = self.parse_age(min_age)
        self.max_age = self.parse_age(max_age)
        self.keywords = keywords
        self.conditions = conditions
        self.title = title
        self.summary = summary
        self.description = description
        self.criteria = criteria

        if self.min_age == -1:
            self.min_age = float("-inf")
        if self.max_age == -1:
            self.max_age = float("inf")

    @staticmethod
    def parse_age(age: str) -> float:
        age_parts = age.split(" ")
        if len(age_parts) != 2 or age == "":
            return -1

        digit = int(age_parts[0])
        unit = age_parts[1]

        if unit.startswith("year"):
            return digit
        elif unit.startswith("month"):
            return digit / 12
        elif unit.startswith("week"):
            return digit / 52
        else:
            return -1

    @staticmethod
    def parse_gender(gender: str) -> str:
        if gender == "":
            return "all"
        else:
            return gender
