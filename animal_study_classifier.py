import re
import logging
from typing import Optional, List, Dict
import aiohttp
import asyncio
from aiolimiter import AsyncLimiter
from transformers import pipeline
import torch
import random

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

        # Cache DOI results
        self.cache: Dict[str, float] = {}
        # Rate limiter to avoid hitting API limits
        self.limiter = AsyncLimiter(max_requests_per_second, 1)
        # Track errors
        self.errors: Dict[str, str] = {}

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
                            return await resp.json()
                        elif 500 <= resp.status < 600:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history,
                                status=resp.status, message="Server error"
                            )
                        else:
                            logging.warning(f"Request to {url} returned status {resp.status}")
                            return None
                except Exception as e:
                    if attempt == max_attempts:
                        logging.error(f"Failed to fetch {url} after {attempt} attempts: {e}")
                        return None
                    else:
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
            return "No abstract available"
        return re.sub(r'</?jats:[^>]+>', '', abstract)

    @staticmethod
    def combine_text(title: str, abstract: str, concepts: List[dict]) -> str:
        concepts_text = ", ".join([f"{c['display_name']} ({c.get('score', 0):.2f})" for c in concepts])
        return f"Title: {title}\nAbstract: {abstract}\nConcepts: {concepts_text}"

    # -------------------- Classification --------------------
    def classify_text(self, text: str) -> float:
        try:
            output = self.classifier(text, self.candidate_labels)
            label_score_dict = dict(zip(output["labels"], output["scores"]))
            return label_score_dict.get(self.target_label, 0.0)
        except Exception as e:
            logging.error(f"Classification failed: {repr(e)}")
            return 0.0

    # -------------------- Main DOI Function --------------------
    async def check_for_valid_animal_study(self, doi: str, session: aiohttp.ClientSession) -> float:
        if doi in self.cache:
            logging.info(f"{doi}: Returning cached result")
            return self.cache[doi]

        try:
            # Try OpenAlex first
            openalex_data = await self.fetch_openalex(session, doi)

            if openalex_data:
                if openalex_data.get('type') == 'review':
                    self.cache[doi] = 0.0
                    logging.info(f"{doi}: Excluded (review)")
                    return 0.0
                title = openalex_data.get('title', "No title available")
                abstract_index = openalex_data.get('abstract_inverted_index')
                abstract = self.reconstruct_abstract(abstract_index)
                if not abstract:  # fallback to Crossref for missing abstract
                    crossref_data = await self.fetch_crossref(session, doi)
                    abstract = self.clean_abstract(crossref_data.get('abstract') if crossref_data else None)
                    if abstract == "No abstract available":
                        self.errors[doi] = "No abstract available"
                concepts = openalex_data.get('concepts', [])

            else:
                # Fallback to Crossref if OpenAlex missing
                crossref_data = await self.fetch_crossref(session, doi)
                if not crossref_data:
                    self.cache[doi] = 0.0
                    self.errors[doi] = "Missing OpenAlex and CrossRef data"
                    return 0.0
                title = crossref_data.get('title', ["No title available"])[0]
                abstract = self.clean_abstract(crossref_data.get('abstract'))
                concepts = []

            combined_text = self.combine_text(title, abstract, concepts)
            score = self.classify_text(combined_text)
            self.cache[doi] = score
            logging.info(f"{doi}: Classification completed, score={score:.2f}")
            return score

        except Exception as e:
            self.errors[doi] = str(e)
            logging.error(f"{doi}: Failed with exception {repr(e)}")
            self.cache[doi] = 0.0
            return 0.0

    # -------------------- Batch Processing --------------------
    async def batch_check(self, doi_list: List[str]) -> Dict[str, float]:
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_for_valid_animal_study(doi, session) for doi in doi_list]
            scores = await asyncio.gather(*tasks)
            logging.info(f"Batch processing completed: {len(doi_list)} DOIs processed")
            if self.errors:
                logging.warning(f"Some DOIs had errors: {len(self.errors)} errors logged")
            return dict(zip(doi_list, scores))
