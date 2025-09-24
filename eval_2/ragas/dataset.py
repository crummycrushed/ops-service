import pandas as pd

class RAGEvaluationDataset:

    def __init__(self):
        self.data = self._create_dataset()


    def _create_dataset(self):
        return {
            "question": [
                "What is photosynthesis and why is it important?",
                "How does machine learning work?",
                "What causes climate change?",
                "What is blokchain technology?"
            ],
            "contexts": [
                ["Photsyntesis is the process by which plan convert light energy into chemical energy..."],
                ["machine learning is a subset of AI enabling computers to learn from data..."],
                ["Climate change refers to long term shifts in global temperature..."],
                ["Blokchain is a distrubuted ledger techbology with cryptographic links..."]
            ],
            "ground_truth":[
                "Photsynthesis converts sunlight, CO2, and water into glucose and oxygen",
                "Machine learning enabled computers to learn fom data wothout explicit programming",
                "Climate change involves global temperature shifts, primarliy caused by humans burning fuels.",
                "Blockchain creates an immutable chain of transaction records using cryptograpgy."

            ]
        }