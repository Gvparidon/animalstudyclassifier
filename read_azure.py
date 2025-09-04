import os
import json
import pandas as pd

class ReadAzure:
    def __init__(self, input_file, azure_dir):
        self.input_file = input_file
        self.azure_dir = azure_dir

    def read_azure_output(self):
        """Reads all JSONL files from azure_dir and returns a dataframe with parsed content."""
        records = []

        # Loop through all jsonl files
        output_dir = os.path.join(self.azure_dir, "output")
        for file in os.listdir(output_dir):
            if file.endswith(".jsonl"):
                file_path = os.path.join(output_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            custom_id = data.get("custom_id")

                            # Navigate to assistant message content
                            message_content = (
                                data.get("response", {})
                                .get("body", {})
                                .get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "{}")
                            )

                            # Parse inner JSON content
                            parsed_content = json.loads(message_content)

                            records.append({
                                "DOI": custom_id,
                                "animal_testing": parsed_content.get("animal_testing"),
                                "in_vivo": parsed_content.get("in_vivo"),
                                "location": parsed_content.get("location"),
                                "species": parsed_content.get("species"),
                            })
                        except Exception as e:
                            print(f"Skipping line due to error: {e}")
        return pd.DataFrame(records)


if __name__ == "__main__":
    input_file = "data/output_20250902_232836.xlsx"
    azure_dir = "azure"

    # Load the Excel file
    df_input = pd.read_excel(input_file)

    # Load and parse Azure outputs
    reader = ReadAzure(input_file, azure_dir)
    df_azure = reader.read_azure_output()

    # Merge on DOI vs custom_id
    df_merged = df_input.merge(df_azure, on="DOI", how="left")

    # To excel  
    df_merged.to_excel("data/final_output/final_output.xlsx", index=False)
    
