# Standard library
import asyncio
import logging
import os
import random
import re
import ssl
from typing import Dict, List, Optional, Tuple

# Third-party libraries
import aiohttp
import certifi
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
import torch
from transformers import pipeline
from dotenv import load_dotenv

# -------------------- API Keys ---------------------
load_dotenv()
ELSEVIER_KEY = os.getenv("ELSEVIER_KEY")

# -------------------- SSL Context --------------------
ssl_context = ssl.create_default_context(cafile=certifi.where())

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------- Class --------------------
class AnimalStudyClassifier:
    def __init__(self, max_requests_per_second: int = 2):
        # Use GPU if available
        self.device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Using device: {'GPU' if self.device==0 else 'CPU'}")

        # Initialize zero-shot classifier
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )

        self.candidate_labels = [
            "in vivo animal study (live animals used in experiments)",
            "not an animal study; purely theoretical, mathematical, or unrelated field"
        ]
        self.target_label = self.candidate_labels[0]

        self.species_list = ['Mice', 'Rats', 'Zebrafish', 'Drosophila', 'Callithrix', 'Gastropoda', 'Hylobates']

        # Cache DOI results
        self.cache: Dict[str, float] = {}
        # Rate limiter to avoid hitting API limits
        self.limiter = AsyncLimiter(max_requests_per_second, 1)
        # Track types and their sources
        self.types: Dict[str, str] = {}
        self.type_sources: Dict[str, str] = {}
        # Track abstracts
        self.abstracts: Dict[str, str] = {}
        # Track titles
        self.titles: Dict[str, str] = {}
        # Track publisher
        self.publisher: Dict[str, str] = {}
        # Track first/last author organization
        self.first_author_org: Dict[str, List[str]] = {}
        self.last_author_org: Dict[str, List[str]] = {}

        # Track animals used
        self.animals_used: Dict[str, bool] = {}
        # Track animal confidence
        self.animal_confidence: Dict[str, float] = {}
        # Track animal evidence terms
        self.animal_evidence_terms: Dict[str, List[str]] = {}

        # Track in vivo
        self.in_vivo: Dict[str, bool] = {}
        # Track in vivo confidence
        self.in_vivo_confidence: Dict[str, float] = {}
        # Track in vivo evidence terms
        self.in_vivo_evidence_terms: Dict[str, List[str]] = {}

        # Track species
        self.species: Dict[str, str] = {}
        # Track species evidence terms
        self.species_evidence_terms: Dict[str, List[str]] = {}

        # Track errors
        self.errors: Dict[str, str] = {}
        
        # Types to exclude (original research filter)
        self.excluded_types = {
            'review', 'conference-review', 'systematic-review', 'meta-analysis',
            'editorial', 'letter', 'commentary', 'correspondence', 'case-report',
            'book-review', 'book-chapter', 'conference-paper', 'preprint', 'erratum'
        }

    # -------------------- Async HTTP with retries --------------------
    async def fetch_json(self, session: aiohttp.ClientSession, url: str, max_attempts=5, base_delay=2, timeout=60) -> Optional[dict]:
        """
        Fetch JSON with retries, exponential backoff, and jitter.
        """
        async with self.limiter:
            for attempt in range(1, max_attempts + 1):
                try:
                    async with session.get(url, timeout=timeout) as resp:
                        if resp.status == 200:
                            try:
                                return await resp.json()
                            except aiohttp.ContentTypeError as e:
                                logging.error(f"Invalid JSON from {url}: {e}")
                                return None
                        elif resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", base_delay))
                            logging.warning(f"Rate limited on {url}, retrying after {retry_after}s...")
                            await asyncio.sleep(retry_after)
                        elif 500 <= resp.status < 600:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history,
                                status=resp.status, message="Server error"
                            )
                        elif 400 <= resp.status < 500:
                            # Permanent client error (except 429), stop retrying
                            logging.warning(f"Permanent client error {resp.status} on {url}, skipping retries.")
                            return None
                        else:
                            logging.warning(f"Unexpected status {resp.status} from {url}")
                            return None

                except Exception as e:
                    if attempt == max_attempts:
                        logging.error(f"Failed to fetch {url} after {attempt} attempts: {e}")
                        return None
                    wait_time = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logging.warning(f"Failed {url} (attempt {attempt}): {e}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

    async def fetch_openalex(self, session: aiohttp.ClientSession, doi: str) -> Optional[dict]:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        return await self.fetch_json(session, url)

    async def fetch_crossref(self, session: aiohttp.ClientSession, doi: str) -> Optional[dict]:
        url = f"https://api.crossref.org/works/{doi}"
        data = await self.fetch_json(session, url)
        if data and "message" in data:
            return data["message"]
        return None

    async def fetch_pubmed_abstract(self, session: aiohttp.ClientSession, pmid_url: str) -> str:
        """
        Fetch abstract from PubMed page HTML (fallback).
        """
        async with self.limiter:
            try:
                async with session.get(pmid_url, timeout=60, ssl=ssl_context) as resp:
                    if resp.status != 200:
                        logging.warning(f"PubMed request failed with status {resp.status}")
                        return None

                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    abstract_div = soup.find("div", {"id": "eng-abstract"})
                    if abstract_div:
                        return abstract_div.get_text(strip=True, separator=" ")
                    else:
                        return None
            except Exception as e:
                logging.error(f"Failed to fetch PubMed abstract from {pmid_url}: {repr(e)}")
                return None

    async def fetch_springer_abstract(self, session: aiohttp.ClientSession, doi: str) -> Optional[str]:
        """
        Try to scrape Springer abstracts directly from the article page.
        """
        url = f"https://doi.org/{doi}"
        try:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, ssl=ssl_context) as resp:
                if resp.status != 200:
                    logging.warning(f"Springer request failed with status {resp.status} for {doi}")
                    return None
                text = await resp.text()
                soup = BeautifulSoup(text, "html.parser")
                abstract_div = soup.find("div", {"class": "c-article-section__content", "id": "Abs1-content"})
                if abstract_div:
                    return abstract_div.get_text(strip=True, separator=" ")
        except Exception as e:
            logging.error(f"Failed Springer abstract fetch for {doi}: {repr(e)}")
        return None

    async def fetch_elsevier_abstract(self, session: aiohttp.ClientSession, doi: str) -> Optional[str]:
        """
        Use Elsevier API to retrieve abstract.
        """
        if not ELSEVIER_KEY:
            logging.error("No Elsevier API key available")
            return None
        
        url = f"https://api.elsevier.com/content/article/doi/{doi}"
        headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": ELSEVIER_KEY,
        }
        try:
            async with session.get(url, headers=headers, ssl=ssl_context) as resp:
                if resp.status != 200:
                    logging.warning(f"Elsevier request failed with status {resp.status} for {doi}")
                    return None
                data = await resp.json()
                return (
                    data.get("full-text-retrieval-response", {})
                        .get("coredata", {})
                        .get("dc:description")
                )
        except Exception as e:
            logging.error(f"Failed Elsevier abstract fetch for {doi}: {repr(e)}")
        return None

    # -------------------- Data Processing --------------------
    @staticmethod
    def reconstruct_abstract(inverted_index: dict) -> str:
        if not inverted_index:
            return ""
        max_index = max(max(indices) for indices in inverted_index.values())
        words = [""] * (max_index + 1)
        for word, indices in inverted_index.items():
            for i in indices:
                words[i] = word
        return " ".join(words)

    @staticmethod
    def clean_abstract(abstract: Optional[str]) -> str:
        if not abstract:
            return None
        return re.sub(r'</?jats:[^>]+>', '', abstract)

    @staticmethod
    def combine_text(title: str, abstract: str, concepts: List[dict]) -> str:
        concepts_text = ", ".join([f"{c['display_name']} ({c.get('score', 0):.2f})" for c in concepts])
        return f"Title: {title}\nAbstract: {abstract}\nConcepts: {concepts_text}"

    # -------------------- Type Filtering --------------------
    def should_exclude_type(self, paper_type: str) -> bool:
        """
        Check if paper type should be excluded (not original research).
        Uses partial matching to catch variations.
        """
        if not paper_type:
            return False
        
        paper_type_lower = paper_type.lower().replace('-', ' ').replace('_', ' ')
        
        for excluded_type in self.excluded_types:
            excluded_lower = excluded_type.lower().replace('-', ' ').replace('_', ' ')
            if excluded_lower in paper_type_lower:
                return True
        
        return False

    # -------------------- Classification --------------------
    def _classify_animals_used(self, mesh_terms: List[Dict]) -> Tuple[bool, str, List[str]]:
        
        # Key MeSH UIs for animal detection
        animal_mesh_uis = {
            "D000818": "Animals",
            "D023421": "Models, Animal", 
            "D004195": "Disease Models, Animal",
            "D032761": "Animal Experimentation"
        }
        
        # Human-only indicator
        human_mesh_ui = "D006801"
        
        # In vitro indicators
        in_vitro_uis = {
            "D066298": "In Vitro Techniques",
            "D002478": "Cells, Cultured",
            "D018929": "Cell Culture Techniques"
        }
        
        found_animal_terms = []
        found_human_terms = []
        found_in_vitro_terms = []
        
        # Check each MeSH term
        for ui,name in mesh_terms.items():
            
            if ui in animal_mesh_uis:
                found_animal_terms.append(f"{name} ({ui})")
            elif ui == human_mesh_ui:
                found_human_terms.append(f"{name} ({ui})")
            elif ui in in_vitro_uis:
                found_in_vitro_terms.append(f"{name} ({ui})")
        
        # Classification logic based on configuration
        animals_used = False
        confidence = "low"
        evidence_terms = []
        
        if found_animal_terms:
            animals_used = True
            evidence_terms.extend(found_animal_terms)
            confidence = "high" if len(found_animal_terms) > 1 else "medium"
        
        # Reduce confidence if strong in vitro indicators present
        if found_in_vitro_terms and confidence in ["high"]:
            confidence = "medium"
        
        return animals_used, confidence, evidence_terms
    
    def _classify_in_vivo(self, mesh_terms: List[Dict], animals_confidence: str) -> Tuple[bool, str, List[str]]:
        
        # Strong in vitro indicators
        in_vitro_uis = {
            "D066298": "In Vitro Techniques",
            "D002478": "Cells, Cultured", 
            "D018929": "Cell Culture Techniques",
            "D046508": "Cell Culture",
            "D019149": "Bioreactors"
        }
        
        # In vivo supporting terms
        in_vivo_supporting_uis = {
            "D032761": "Animal Experimentation",
            "D023421": "Models, Animal",
            "D004195": "Disease Models, Animal",
            "D001522": "Behavioral Phenomena",
            "D023041": "Xenograft Model Antitumor Assays"
        }
        
        found_in_vitro_terms = []
        found_in_vivo_terms = []
        
        for ui, name in mesh_terms.items():
            if ui in in_vitro_uis:
                found_in_vitro_terms.append(f"{name} ({ui})")
            elif ui in in_vivo_supporting_uis:
                found_in_vivo_terms.append(f"{name} ({ui})")
        
        # Classification logic
        if found_in_vitro_terms and not found_in_vivo_terms:
            # Strong in vitro indicators, no in vivo support
            in_vivo = False
            confidence = "medium"
            evidence_terms = found_in_vitro_terms
        
        elif found_in_vivo_terms:
            # Strong in vivo support
            in_vivo = True
            confidence = "high"
            evidence_terms = found_in_vivo_terms
        
        else:
            # Default assumption: if animals used, likely in vivo
            in_vivo = True
            confidence = "low" if animals_confidence == "low" else "medium"
            evidence_terms = ["Assumption: animals present without strong in vitro indicators"]
        
        return in_vivo, confidence, evidence_terms
    
    def _extract_species(self, mesh_terms: List[Dict]) -> Tuple[List[str], List[str]]:
        
        # Common species mapping (simplified for now)
        species_mapping = {
            "D051379": "Mice",
            "D051381": "Rats", 
            "D011817": "Rabbits",

        }
        
        found_species = []
        evidence_terms = []
        
        for ui,name in mesh_terms.items():
            
            if ui in species_mapping:
                species_name = species_mapping[ui]
                if species_name not in found_species:
                    found_species.append(species_name)
                    evidence_terms.append(f"{species_name} ({ui})")
        
        return found_species, evidence_terms


    def classify_text(self, text: str) -> float:
        try:
            output = self.classifier(text, self.candidate_labels)
            label_score_dict = dict(zip(output["labels"], output["scores"]))
            target_score = label_score_dict.get(self.target_label, 0.0)
            return target_score
        except Exception as e:
            logging.error(f"Classification failed: {repr(e)}")
            return 0.0

    # -------------------- Main DOI Function --------------------
    async def check_for_valid_animal_study(self, doi: str, session: aiohttp.ClientSession) -> float:
        if doi in self.cache:
            return self.cache[doi]

        try:
            openalex_data = await self.fetch_openalex(session, doi)

            if not openalex_data:
                # OpenAlex missing
                self.cache[doi] = 0.0
                self.errors[doi] = "Missing OpenAlex data"
                self.titles[doi] = None
                self.abstracts[doi] = None
                return 0.0

            paper_type = openalex_data.get('type', None)
            self.types[doi] = paper_type
            self.type_sources[doi] = "OpenAlex"

            if self.should_exclude_type(paper_type):
                self.cache[doi] = 0.0
                self.titles[doi] = openalex_data.get('title', None)
                self.abstracts[doi] = "Excluded paper type"
                return 0.0

            # Extract title
            self.titles[doi] = openalex_data.get('title', None)

            # Extract mesh terms
            mesh_terms = {
                m['descriptor_ui']: m['descriptor_name']
                for m in openalex_data.get('mesh', [])
                if 'descriptor_ui' in m and 'descriptor_name' in m
            }

            # Check for animal study
            animals_used, confidence, evidence_terms = self._classify_animals_used(mesh_terms)
            self.animals_used[doi] = animals_used
            self.animal_confidence[doi] = confidence
            self.animal_evidence_terms[doi] = evidence_terms

            # Check for in vivo/species
            if animals_used:
                in_vivo, confidence, in_vivo_evidence_terms = self._classify_in_vivo(mesh_terms, confidence)
                species, species_evidence_terms = self._extract_species(mesh_terms)
                self.in_vivo[doi] = in_vivo
                self.in_vivo_confidence[doi] = confidence
                self.in_vivo_evidence_terms[doi] = in_vivo_evidence_terms
                self.species[doi] = species
                self.species_evidence_terms[doi] = species_evidence_terms

            # Publisher
            self.publisher[doi] = openalex_data.get("primary_location", {}).get("source", {}).get("host_organization_name", None)

            # Abstract fetching (PubMed -> CrossRef -> OpenAlex -> Publisher-specific)
            abstract = None
            try:
                pmid_url = openalex_data['ids'].get('pmid')
                if pmid_url:
                    abstract = await self.fetch_pubmed_abstract(session, pmid_url)
                if not abstract:
                    crossref_data = await self.fetch_crossref(session, doi)
                    abstract = self.clean_abstract(crossref_data.get('abstract') if crossref_data else None)
                if not abstract:
                    abstract = self.reconstruct_abstract(openalex_data.get('abstract_inverted_index'))
                if not abstract:
                    publisher = self.publisher.get(doi, "")
                    if publisher in ["Springer Science+Business Media"]:
                        abstract = await self.fetch_springer_abstract(session, doi)
                    elif publisher == "Elsevier BV":
                        abstract = await self.fetch_elsevier_abstract(session, doi)
            except Exception as e:
                logging.error(f"Failed to fetch abstract for {doi}: {repr(e)}")
                self.errors[doi] = "Failed to fetch abstract"
                abstract = None

            self.abstracts[doi] = abstract
            concepts = openalex_data.get('concepts', [])

            # Author institutions
            self.first_author_org[doi] = [inst.get('display_name', None) for inst in openalex_data.get('authorships', [{}])[0].get('institutions', [{"display_name": None}])]
            self.last_author_org[doi] = [inst.get('display_name', None) for inst in openalex_data.get('authorships', [{}])[-1].get('institutions', [{"display_name": None}])]

            combined_text = self.combine_text(self.titles[doi], abstract, concepts)
            score = self.classify_text(combined_text)
            self.cache[doi] = score

            return score

        except Exception as e:
            self.errors[doi] = str(e)
            logging.error(f"{doi}: Failed with exception {repr(e)}")
            self.cache[doi] = 0.0
            return 0.0


    # -------------------- Batch Processing --------------------
    async def batch_check(self, doi_list: List[str]) -> Dict[str, float]:
        async with aiohttp.ClientSession() as session:
            # Create progress bar for batch processing
            pbar = tqdm(total=len(doi_list), desc="Processing papers", unit="paper")
            
            async def process_with_progress(doi):
                result = await self.check_for_valid_animal_study(doi, session)
                pbar.update(1)
                return result
            
            tasks = [process_with_progress(doi) for doi in doi_list]
            scores = await asyncio.gather(*tasks)
            pbar.close()
            
            logging.info(f"Batch processing completed: {len(doi_list)} DOIs processed")
            if self.errors:
                logging.warning(f"Some DOIs had errors: {len(self.errors)} errors logged")
            return dict(zip(doi_list, scores))



if __name__ == "__main__":

    async def main():
        classifier = AnimalStudyClassifier()
        import pandas as pd
        #df = pd.read_excel('data/output_20250902_232836 - Copy.xlsx')
        #df = df[df.Abstract == 'No abstract available']
        #df = df[df["DOI"].notna() & (df["DOI"] != "")]
        doi_list = ['10.1007/S10841-023-00537-0', '10.1007/S10344-023-01760-5', '10.1016/J.AGRFORMET.2024.110179']

        await classifier.batch_check(doi_list)

        # Print abstracts
        for doi in doi_list:
            abstract = classifier.abstracts.get(doi, None)
            print(doi + ": " + abstract)

    # Properly run the async main
    asyncio.run(main())
