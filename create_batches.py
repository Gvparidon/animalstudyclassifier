# create_batches.py
import pandas as pd
import os
import json
import yaml
from datetime import datetime
from pdf_scraper import PDFScraper

class BatchCreator:
    def __init__(self, input_file, azure_dir, batch_size=100):
        self.input_file = input_file
        self.azure_dir = azure_dir
        self.batch_size = batch_size
        self.data = input_file

    def _load_prompt(self):
        with open(os.path.join(self.azure_dir, f"prompt.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def create_batches(self):
        prompt = self._load_prompt()
        scraper = PDFScraper()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_file = os.path.join(self.azure_dir, 'batches', f"batch_{timestamp}.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for i, doi in enumerate(self.data['DOI']):

                full_text = scraper.get_full_text(doi)

                task_obj = {
                    "custom_id": f"task-{i}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": full_text}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "AnimalStudyResponse",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "animal_testing": {
                                            "type": "string",
                                            "enum": ["yes", "no"]
                                        },
                                        "in_vivo": {
                                            "type": "string",
                                            "enum": ["yes", "no"]
                                        },
                                        "location": {
                                            "type": "string"
                                        },
                                        "species": {
                                            "type": "string"
                                        }
                                    },
                                    "required": ["animal_testing", "in_vivo"],
                                    "additionalProperties": False,
                                    "if": {
                                        "properties": {"animal_testing": {"const": "yes"}, "in_vivo": {"const": "yes"}}
                                    },
                                    "then": {
                                        "required": ["location", "species"]
                                    }
                                }
                            }
                        }
                    }
                }

                f.write(json.dumps(task_obj) + "\n")

        print(f"JSONL file created at: {output_file}")
        scraper.close()
        return output_file


if __name__ == "__main__":
    input_file = "data/output.xlsx"
    azure_dir = "azure"
    batch_size = 100
    df = pd.read_excel(input_file)
    df = df.head(2)

    
    batch_creator = BatchCreator(df, azure_dir, batch_size)
    batch_creator.create_batches()
