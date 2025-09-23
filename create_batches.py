# create_batches.py
import pandas as pd
import os
import json
import yaml
from datetime import datetime
from text_fetcher import PaperFetcher

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

        fetcher = PaperFetcher()

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

            missing_file = os.path.join(batch_dir, f"missing_methods_{timestamp}.txt")

            with open(output_file, "w", encoding="utf-8") as f, open(missing_file, "a", encoding="utf-8") as mf:
                for doi in batch:
                    full_text = fetcher.fetch_full_paper_text(
                        doi,
                        self.data.loc[self.data['DOI'] == doi, 'Title'].values[0],
                        self.data.loc[self.data['DOI'] == doi, 'Publisher'].values[0]
                    )
                    try:
                        method_section = fetcher.extract_methods_text(full_text.sections)
                        ethics_section = fetcher.extract_ethics_text(full_text.sections)
                    except:
                        method_section = ""
                        ethics_section = ""

                    # If no method section → log DOI separately
                    if not method_section or method_section.strip() == "":
                        mf.write(f"{doi}\n")
                        continue

                    
                    abstract = self.data.loc[self.data['DOI'] == doi, 'Abstract'].values[0]

                    content = f"Abstract: {abstract}\nMethod section: {method_section}\n {ethics_section}"

                    task_obj = {
                        "custom_id": doi,
                        "method": "POST",
                        "url": "/chat/completions",
                        "body": {
                            "model": "gpt-4.1",
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": content}
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
    input_file = "data/output_20250911_152540.xlsx"
    azure_dir = "azure"
    batch_size = 1500
    df = pd.read_excel(input_file)
    
    df = df[~df.duplicated('DOI')]
    print(len(df))
    df = df[(df.BART_MNLI_Score >= 0.7) | (df.Animals_Used == True)]
    print(len(df))

    #batch_creator = BatchCreator(df, azure_dir, batch_size)
    #batch_creator.create_batches()
