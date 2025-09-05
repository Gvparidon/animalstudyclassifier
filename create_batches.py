# create_batches.py
import pandas as pd
import os
import json
import yaml
from datetime import datetime
from pmc_text_fetcher import PMCTextFetcher
from ubn_text_fetcher import UBNTextFetcher

class BatchCreator:
    def __init__(self, input_file, azure_dir, batch_size=100):
        self.input_file = input_file
        self.azure_dir = azure_dir
        self.batch_size = batch_size
        self.data = input_file

    def _load_prompt(self):
        with open(os.path.join(self.azure_dir, f"prompt.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)['prompt']

    def create_batches(self):
        prompt = self._load_prompt()
        # scraper = PDFScraper()
        fetcher = PMCTextFetcher()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        batch_dir = os.path.join(self.azure_dir, 'batches')
        os.makedirs(batch_dir, exist_ok=True)

        dois = self.data['DOI'].tolist()

        batch_files = []
        for batch_idx in range(0, len(dois), self.batch_size):
            batch = dois[batch_idx: batch_idx + self.batch_size]
            output_file = os.path.join(
                batch_dir,
                f"batch_{timestamp}_{batch_idx//self.batch_size + 1}.jsonl"
            )

            with open(output_file, "w", encoding="utf-8") as f:
                for doi in batch:
                    # full_text = scraper.get_full_text(doi)
                    method_section = fetcher.fetch_methods_text(doi)
                    if method_section == '':
                        continue

                    task_obj = {
                        "custom_id": doi,
                        "method": "POST",
                        "url": "/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": method_section}
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
                                            },
                                            "approving_organization": {
                                                "type": "string"
                                            }
                                        },
                                        "required": ["animal_testing", "in_vivo", "location", "species", "approving_organization"],
                                        "additionalProperties": False
                                    }
                                }
                            }
                        }
                    }

                    f.write(json.dumps(task_obj) + "\n")

            print(f"Batch file created at: {output_file}")
            batch_files.append(output_file)

        return batch_files


if __name__ == "__main__":
    input_file = "data/output_20250902_232836.xlsx"
    azure_dir = "azure"
    batch_size = 60
    df = pd.read_excel(input_file)

    df = df[(df.BART_MNLI_Score >= 0.7) | (df.Mesh_Term == True)]
    df = df[~df.duplicated(subset='DOI')]

    batch_creator = BatchCreator(df, azure_dir, batch_size)
    batch_creator.create_batches()
