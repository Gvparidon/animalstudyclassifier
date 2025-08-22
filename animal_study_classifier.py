import re
import logging
from typing import Optional, List, Dict
import aiohttp
import asyncio
from aiolimiter import AsyncLimiter
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnimalStudyClassifier:
    def __init__(self, max_requests_per_second: int = 5):
        """
        max_requests_per_second: maximum allowed requests per second to APIs
        """
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")
        self.candidate_labels = [
            "in vivo animal study (live animals used in experiments)",
            "not an animal study; purely theoretical, mathematical, or unrelated field"
        ]
        self.target_label = self.candidate_labels[0]

        # Caching DOI results to avoid repeated requests
        self.cache: Dict[str, float] = {}

        # Rate limiter to avoid hitting API limits
        self.limiter = AsyncLimiter(max_requests_per_second, 1)  # max N requests per second

        # Track DOI processing status
        self.errors: Dict[str, str] = {}  # DOI -> error message

    # ------------------ Async Fetching ------------------ #
    async def fetch_json(self, session: aiohttp.ClientSession, url: str) -> Optional[dict]:
        async with self.limiter:  # enforce rate limit
            for attempt in range(3):  # retry up to 3 times
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logging.warning(f"Request to {url} returned status {response.status}")
                            await asyncio.sleep(1)
                except Exception as e:
                    logging.warning(f"Failed to fetch {url} (attempt {attempt+1}): {e}")
                    await asyncio.sleep(1)
            return None

    async def fetch_openalex(self, session: aiohttp.ClientSession, doi: str) -> Optional[dict]:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        return await self.fetch_json(session, url)

    async def fetch_crossref(self, session: aiohttp.ClientSession, doi: str) -> Optional[dict]:
        url = f"https://api.crossref.org/works/{doi}"
        data = await self.fetch_json(session, url)
        return data.get('message') if data else None

    # ------------------ Data Processing ------------------ #
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
    def clean_abstract(abstract: str) -> str:
        if not abstract:
            return "No abstract available"
        return re.sub(r'</?jats:[^>]+>', '', abstract)

    @staticmethod
    def combine_text(title: str, abstract: str, concepts: List[dict]) -> str:
        concepts_text = ", ".join([f"{c['display_name']} ({c['score']:.2f})" for c in concepts])
        return f"Title: {title}\nAbstract: {abstract}\nConcepts: {concepts_text}"

    # ------------------ Classification ------------------ #
    def classify_text(self, text: str) -> float:
        output = self.classifier(text, self.candidate_labels)
        label_score_dict = dict(zip(output["labels"], output["scores"]))
        return label_score_dict.get(self.target_label, 0.0)

    # ------------------ Main DOI Function ------------------ #
    async def check_for_valid_animal_study(self, doi: str, session: aiohttp.ClientSession) -> float:
        if doi in self.cache:
            logging.info(f"{doi}: Returning cached result")
            return self.cache[doi]

        try:
            openalex_data = await self.fetch_openalex(session, doi)
            crossref_data = None

            if not openalex_data:
                # fallback to CrossRef
                crossref_data = await self.fetch_crossref(session, doi)
                if not crossref_data:
                    self.cache[doi] = 0.0
                    self.errors[doi] = "Missing OpenAlex and CrossRef data"
                    logging.warning(f"{doi}: Missing OpenAlex and CrossRef data")
                    return 0.0
                title = crossref_data.get('title', ["No title available"])[0]
                abstract = self.clean_abstract(crossref_data.get('abstract', None))
                concepts = []
            else:
                if openalex_data.get('type') == 'review':
                    logging.info(f"{doi}: Excluded (review)")
                    self.cache[doi] = 0.0
                    return 0.0

                title = openalex_data.get('title', "No title available")
                abstract_index = openalex_data.get('abstract_inverted_index')
                abstract = self.reconstruct_abstract(abstract_index)

                if not abstract:
                    crossref_data = await self.fetch_crossref(session, doi)
                    abstract = self.clean_abstract(crossref_data.get('abstract', None) if crossref_data else None)

                concepts = openalex_data.get('concepts', [])

            combined_text = self.combine_text(title, abstract, concepts)
            score = self.classify_text(combined_text)

            self.cache[doi] = score
            logging.info(f"{doi}: Classification completed, score={score:.2f}")
            return score

        except Exception as e:
            self.errors[doi] = str(e)
            logging.error(f"{doi}: Failed with exception {e}")
            self.cache[doi] = 0.0
            return 0.0

    # ------------------ Batch Processing ------------------ #
    async def batch_check(self, doi_list: List[str]) -> Dict[str, float]:
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_for_valid_animal_study(doi, session) for doi in doi_list]
            scores = await asyncio.gather(*tasks)
            logging.info(f"Batch processing completed: {len(doi_list)} DOIs processed")
            if self.errors:
                logging.warning(f"Some DOIs had errors: {len(self.errors)} errors logged")
            return dict(zip(doi_list, scores))


