import asyncio
import pandas as pd
from animal_study_classifier import AnimalStudyClassifier
import time
import random
from datetime import datetime
from tqdm import tqdm

async def main():
    start_time = time.time()

    all_dois = pd.read_excel("data/publicaties.xlsx")["DOI nummer"].tolist()
    random.seed(42)
    dois = random.sample(all_dois, min(5, len(all_dois)))

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
        abstract = classifier.abstracts.get(doi, "No abstract available")
        title = classifier.titles.get(doi, "No title available")
        
        # Get in vivo analysis
        in_vivo_analysis = classifier.in_vivo_results.get(doi, {})
        
        # Get ethics analysis
        ethics_analysis = classifier.ethics_results.get(doi, {})
        
        results_data.append({
            "DOI": doi,
            "Title": title,
            "BART_MNLI_Score": score,
            "Paper_Type": paper_type,
            "Type_Source": type_source,
            "Abstract": abstract,
            "Species_Detected": ", ".join(in_vivo_analysis.get("species_detected", [])),
            "Species_Sentences": " | ".join(in_vivo_analysis.get("species_sentences", [])),
            "In_Vivo_Keywords": ", ".join(in_vivo_analysis.get("in_vivo_keywords", [])),
            "In_Vivo_Sentences": " | ".join(in_vivo_analysis.get("evidence_sentences", [])),
            "Ethics_Institutions": ", ".join(ethics_analysis.get("institutions_detected", [])),
            "Ethics_Institution_Sentences": " | ".join(ethics_analysis.get("institution_sentences", [])),
            "Ethics_Keywords": ", ".join(ethics_analysis.get("ethics_keywords", [])),
            "Ethics_Sentences": " | ".join(ethics_analysis.get("evidence_sentences", [])),
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