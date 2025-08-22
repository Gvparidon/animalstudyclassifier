import asyncio
import pandas as pd
from animal_study_classifier import AnimalStudyClassifier
import time

async def main():
    start_time = time.time()

    # Load DOIs from file
    dois = pd.read_excel("data/publicaties.xlsx")["DOI nummer"].tolist()
    dois = dois[:50]

    # Initialize classifier
    classifier = AnimalStudyClassifier(max_requests_per_second=10)

    # Run batch classification
    results = await classifier.batch_check(dois)

    # Convert results to DataFrame
    df_results = pd.DataFrame(list(results.items()), columns=["DOI", "Score"])

    # Save to Excel
    output_file = "data/output.xlsx"
    df_results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Optionally save errors to a separate sheet
    if classifier.errors:
        df_errors = pd.DataFrame(list(classifier.errors.items()), columns=["DOI", "Error"])
        with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
            df_errors.to_excel(writer, sheet_name="Errors", index=False)
        print(f"Errors saved to 'Errors' sheet in {output_file}")

    # Print total runtime
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

# Run the async main
asyncio.run(main())