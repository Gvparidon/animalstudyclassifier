import asyncio
import pandas as pd
from animal_study_classifier import AnimalStudyClassifier
import time
import random

async def main():
    start_time = time.time()

    all_dois = pd.read_excel("data/publicaties.xlsx")["DOI nummer"].tolist()
    random.seed(42)
    dois = random.sample(all_dois, min(10, len(all_dois)))

    classifier = AnimalStudyClassifier()
    results = await classifier.batch_check(dois)

    df_results = pd.DataFrame(list(results.items()), columns=["DOI", "Score"])
    output_file = "data/output_test.xlsx"
    df_results.to_excel(output_file, index=False, sheet_name="Scores")

    if classifier.errors:
        df_errors = pd.DataFrame(list(classifier.errors.items()), columns=["DOI", "Error"])
        with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
            df_errors.to_excel(writer, sheet_name="Errors", index=False)

    if classifier.types:
        df_types = pd.DataFrame(list(classifier.types.items()), columns=["DOI", "Type"])
        # Add source information
        df_types["Source"] = [classifier.type_sources.get(doi, "Unknown") for doi in df_types["DOI"]]
        with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
            df_types.to_excel(writer, sheet_name="Types", index=False)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_file}")

# ------------------- SAFE ENTRY POINT -------------------
if __name__ == "__main__":
    asyncio.run(main())