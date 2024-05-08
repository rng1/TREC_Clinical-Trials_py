import os
import xml.etree.ElementTree as ET
import multiprocessing
import pandas as pd
import re

from data import Trial


class TrialParser:
    CSV_PATH = "C:/Users/rnara/Desktop/TFG/[py] TREC_Clinical-Trials/trials.csv"
    DATA_PATH = "C:/Users/rnara/Desktop/TFG/data/trials"

    def __init__(self):
        if os.path.exists(self.CSV_PATH):
            self.df = self.load_trials_from_csv()
        else:
            self.df = self.load_trials_from_xml()

    def load_trials_from_csv(self) -> pd.DataFrame:
        print("Loading trials from CSV...")
        return pd.read_csv(self.CSV_PATH, sep=";")

    def load_trials_from_xml(self) -> pd.DataFrame:
        print("Loading trials from XML...")

        xml_files = [os.path.join(dirpath, f)
                     for dirpath, dirnames, filenames in os.walk(self.DATA_PATH)
                     for f in filenames if f.endswith(".xml")]

        with multiprocessing.Pool() as pool:
            trials = pool.map(self.process_trial, xml_files)

        df = pd.DataFrame.from_records([trial.__dict__ for trial in trials])
        df.to_csv(self.CSV_PATH, sep=';', index=False)

        return df

    def process_trial(self, file_path: str) -> Trial:
        tree = ET.parse(file_path)
        root = tree.getroot()

        nct_id = root.find(".//nct_id").text.lower()
        print(f"Processing trial {nct_id}...")
        return self.parse_trial(root, nct_id)

    def parse_trial(self, root: ET.Element, nct_id: str) -> Trial:
        title = self.get_element_text(root, ".//brief_title")
        summary = (self.get_element_text(root, ".//brief_summary//textblock")
                   .replace("\n", "").replace("\r", ""))
        description = (self.get_element_text(root, ".//detailed_description//textblock")
                       .replace("\n", "").replace("\r", ""))
        criteria = (self.get_element_text(root, ".//criteria//textblock")
                    .replace("\n", "").replace("\r", ""))
        gender = self.get_element_text(root, ".//gender")
        min_age = self.get_element_text(root, ".//minimum_age")
        max_age = self.get_element_text(root, ".//maximum_age")

        keyword_elements = root.findall(".//keyword")
        keywords = [keyword.text for keyword in keyword_elements]

        mesh_term_elements = root.findall(".//mesh_term")
        mesh_terms = [mesh_term.text for mesh_term in mesh_term_elements]

        condition_elements = root.findall(".//condition")
        conditions = [condition.text for condition in condition_elements]

        # Compress multiple whitespaces into one
        summary = re.sub(" +", " ", summary)
        description = re.sub(" +", " ", description)
        criteria = re.sub(" +", " ", criteria)

        return Trial(
            nct_id=nct_id,
            gender=gender,
            min_age=min_age,
            max_age=max_age,
            keywords=keywords + mesh_terms,
            conditions=conditions,
            title=title,
            summary=summary,
            description=description,
            criteria=criteria
        )

    @staticmethod
    def get_element_text(root: ET.Element, element_path: str) -> str:
        element = root.find(element_path)
        if element is not None:
            return element.text.lower()
        else:
            return ""
