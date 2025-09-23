import ast
import re
import pandas as pd
import numpy as np


TARGET_ORGS = {
    "Radboud University Nijmegen",
    "Radboud University Medical Center",
    "Radboud Institute for Molecular Life Sciences",
}


def to_list_safe(x):
    """Convert stringified list to Python list safely."""
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def org_check(orgs):
    """Return 1 if any organization is in the target set, else 0."""
    return int(any(org in TARGET_ORGS for org in orgs))


def score_results(data: pd.DataFrame) -> pd.DataFrame:
    """Compute classification scores for each row in the dataset."""
    output = pd.DataFrame({"DOI": data["DOI"]})

    # Start with MNLI score
    output["Score"] = data["BART_MNLI_Score"]
    output["Count"] = (data["BART_MNLI_Score"] >= 0.7).astype(int)

    # First / last author org check
    output["First_author"] = data["First_Author_Organization"].apply(
        lambda x: org_check(to_list_safe(x)) > 0
    )
    output["Last_author"] = data["Last_Author_Organization"].apply(
        lambda x: org_check(to_list_safe(x)) > 0
    )
    output["Count"] += output["First_author"].astype(int) + output["Last_author"].astype(int)

    # MeSH: Animals used / in vivo
    output["Animals_Used_MesH"] = data["Animals_Used"].astype(bool)
    output["In_Vivo_MesH"] = data["In_Vivo"].astype(bool)
    output["Count"] += output["Animals_Used_MesH"].astype(int) + output["In_Vivo_MesH"].astype(int)

    # GPT annotations
    output["Animals_Used_GPT"] = (data["animal_testing"] == "yes")
    output["In_Vivo_GPT"] = (data["in_vivo"] == "yes")
    output["Count"] += output["Animals_Used_GPT"].astype(int) + output["In_Vivo_GPT"].astype(int)

    # Location
    loc_match = data["location"].str.contains("radboud|nijmegen", case=False, na=False)
    output["Location_Radboud"] = loc_match
    output["Location"] = data["location"]
    output["Count"] += loc_match.astype(int)

    # Approving organization
    org_match = data["approving_organization"].str.contains(
        "radboud|nijmegen|netherlands", case=False, na=False
    )
    output["Apr_org_netherlands"] = org_match
    output["Approving_organization"] = data["approving_organization"]
    output["Count"] += org_match.astype(int)

    # Extra info
    output["Species"] = data["species"]

    return output


def evaluate_row(data):
    if data['Count'] == 9:
        return True
    elif not data['Animals_Used_MesH'] and data['Score'] < 0.7:
        return False
    elif not data['In_Vivo_GPT'] or not data['Animals_Used_GPT']:
        return False
    elif data.Species == 'infant (human baby)':
        return False
    elif data['Count'] == 8 and data['Location'] == 'No location mentioned':
        return True
    elif not data['Apr_org_netherlands'] and not data['Approving_organization'] == 'No approval mentioned':
        return False
    elif data['First_author'] and data['Last_author'] and data['Apr_org_netherlands'] and data['Location'] in ['The Hague, The Netherlands', 'The Netherlands']:
        return True
    elif (not data['First_author'] and not data['Last_author'] and 
          data['Location'] == 'No location mentioned' and 
          data['Approving_organization'] == 'No approval mentioned'):
        return False
    elif (not data['Location_Radboud'] and not data.Location == 'No location mentioned'):
        return False
    elif (not data['Apr_org_netherlands'] and not data.Approving_organization == 'No approval mentioned'):
        return False
    elif data.Count == 7 and not data.Animals_Used_MesH and not data.In_Vivo_MesH:
        return True
    elif data.First_author and data.Last_author and data.Apr_org_netherlands and data.Location == 'No location mentioned':
        return True
    elif data.Count == 8 and data.Score < 0.7:
        return True
    elif data.Location_Radboud and data.Apr_org_netherlands:
        return True
    elif not data.First_author and not data.Last_author and not data.Location_Radboud:
        return False
    elif re.search(r'radboud|nijmegen', data.Approving_organization, re.IGNORECASE):
        return True
    elif data.First_author and data.Last_author and data.Location_Radboud and data.Approving_organization == 'No approval mentioned':
        return True
    elif data.First_author and data.Last_author and data.Location == 'No location mentioned' and data.Approving_organization == 'No approval mentioned':
        return True
    elif data.First_author and not data.Last_author and data.Location == 'No location mentioned' and data.Approving_organization == 'No approval mentioned':
        return True
    elif not data.First_author and data.Last_author and data.Location == 'No location mentioned' and data.Approving_organization == 'No approval mentioned':
        return False
    elif data.First_author and data.Location_Radboud:
        return True
    elif not data.First_author and not data.Last_author:
        return False
    elif not data.First_author and not data.Location_Radboud:
        return False
    elif data.Location_Radboud:
        return True
    elif data.First_author and data.Apr_org_netherlands:
        return True
    else:
        return 99  # or some default value


def modify_for_tableau(data: pd.DataFrame) -> pd.DataFrame:
    # Auteur column (vectorized conditions)
    conditions = [
        data["First_author"] & data["Last_author"],
        data["First_author"],
        data["Last_author"],
    ]
    choices = [
        "Eerste en laatste auteur",
        "Eerste auteur",
        "Laatste auteur",
    ]
    data["Auteur"] = np.select(conditions, choices, default="Geen van beide")

    # Split on semicolon, expand into lists
    data["Species"] = data["Species"].str.split(r"\s*;\s*")

    # Explode into multiple rows
    data = data.explode("Species", ignore_index=True)

    # Species mapping
    species_mapping = pd.read_excel("species_mapping.xlsx")

    # Example: assume mapping file has "Species" and "Common_Name"
    mapping_dict = species_mapping.set_index("Species")["Standardized Name"].to_dict()
    data["Species"] = data["Species"].map(mapping_dict).fillna(data["Species"])

    data.loc[data["Evaluation"] != True, "Species"] = pd.NA

    ## Left join publicaties
    publicaties = pd.read_excel('data/publicaties.xlsx')

        # Perform left join on DOI
    data = data.merge(
        publicaties[["DOI nummer", "Faculteit", "Onderzoeksinstituut", "Jaar uitgave"]],
        how="left",
        left_on="DOI",
        right_on="DOI nummer"
    )

    # Drop duplicate key column if you don’t need both
    data = data.drop(columns=["DOI nummer"])

    # Remove rows with score 0
    data = data[data.Score != 0]

    # Weghalen lege onderzoeksinst
    data = data[data['Onderzoeksinstituut'].notna() & (data['Onderzoeksinstituut'] != "")]

    return data


if __name__ == "__main__":
    df = pd.read_excel("data/final_output/final_output2.xlsx")
    results = score_results(df)
    results["Evaluation"] = results.apply(evaluate_row, axis=1)
    results = modify_for_tableau(results)
    results.to_excel("data/final_output/Animal_classification.xlsx", index=False)
