import os.path
import xml.etree.ElementTree as ET
import pandas as pd

from data import Topic


class TopicParser:
    """
    Get the topic data.
    If the CSV file is not found, the topics are parsed from the XML file and embeddings are generated.
    """
    CSV_PATH = "C:/Users/rnara/Desktop/TFG/[py] TREC_Clinical-Trials/topics.csv"
    XML_PATH = "C:/Users/rnara/Desktop/TFG/data/topics2022.xml"

    def __init__(self):
        if os.path.exists(self.CSV_PATH):
            self.df = self.load_topics_from_csv()
        else:
            self.df = self.load_topics_from_xml()

    def load_topics_from_csv(self) -> pd.DataFrame:
        print("Loading topics from CSV...")
        return pd.read_csv(self.CSV_PATH, sep=";")

    def load_topics_from_xml(self) -> pd.DataFrame:
        print("Loading topics from XML...")
        tree = ET.parse(self.XML_PATH)
        root = tree.getroot()
        data = []

        for topic in root:
            data_row = self.process_topic(topic)
            data.append(data_row)

        columns = ["id", "age", "gender", "description"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.CSV_PATH, sep=";", index=False)
        return df

    @staticmethod
    def process_topic(topic: ET.Element) -> list:
        new_topic = Topic(
            topic_id=int(topic.attrib.get("number")),
            description=topic.text.lower().lstrip("\n").rstrip("\n")
        )
        data_row = [new_topic.topic_id, new_topic.age, new_topic.gender, new_topic.description]
        return data_row
