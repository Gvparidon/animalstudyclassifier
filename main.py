import asyncio
import pandas as pd
from animal_study_classifier import AnimalStudyClassifier
import time
import random
from datetime import datetime

async def main():
    start_time = time.time()

    # Load publication data
    df = pd.read_excel("data/publicaties.xlsx")

    # Filter out rows with missing or empty DOI numbers
    df = df[df["DOI nummer"].notna() & (df["DOI nummer"].str.strip() != "")]

    # Extract unique DOIs
    unique_dois = df["DOI nummer"].drop_duplicates().tolist()

    # Set random seed for reproducibility and sample a subset of DOIs
    random.seed(422)
    dois = random.sample(unique_dois, 40)
    #dois = unique_dois

    classifier = AnimalStudyClassifier()
    
    try:
        results = await classifier.batch_check(dois)
    except KeyboardInterrupt:
        print("\nProcess interrupted! Saving partial results...")
        results = classifier.cache  # Use cached results
    except Exception as e:
        print(f"\nError during processing: {e}")
        results = classifier.cache  # Use cached results

    # Create timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/output_{timestamp}.xlsx"
    
    # Create comprehensive results DataFrame
    results_data = []
    for doi in dois:
        score = results.get(doi, 0.0)
        paper_type = classifier.types.get(doi, "Unknown")
            
        type_source = classifier.type_sources.get(doi, "Unknown")
        abstract = classifier.abstracts.get(doi, None)
        title = classifier.titles.get(doi, "No title available")
        mesh_term = classifier.mesh_terms.get(doi, False)
        species = classifier.species.get(doi, "")
        publisher = classifier.publisher.get(doi, "")

        first_author_org = classifier.first_author_org.get(doi, "Unknown")
        last_author_org = classifier.last_author_org.get(doi, "Unknown")

        
        results_data.append({
            "DOI": doi,
            "Title": title,
            "BART_MNLI_Score": score,
            "Paper_Type": paper_type,
            "Type_Source": type_source,
            "Abstract": abstract,
            "Mesh_Term": mesh_term,
            "Species": species,
            "First_Author_Organization": first_author_org,
            "Last_Author_Organization": last_author_org,
            "Publisher": publisher,
            "Processing_Status": "Success" if doi not in classifier.errors else "Error",
            "Error_Message": classifier.errors.get(doi, "")
        })
    
    df_results = pd.DataFrame(results_data)
    df_results.to_excel(output_file, index=False, sheet_name="Animal_Study_Classification")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_file}")

# ------------------- SAFE ENTRY POINT -------------------
if __name__ == "__main__":
    asyncio.run(main())