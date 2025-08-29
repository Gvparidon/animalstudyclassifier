import re
import time
import logging
from io import BytesIO
from typing import Optional

import requests
from PyPDF2 import PdfReader
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PDFScraper:
    """Scraper for downloading and extracting text from PDFs in the UBN repository."""

    def __init__(self, wait_time: int = 10, retries: int = 3, backoff: int = 2):
        """
        Initialize the scraper.

        Args:
            wait_time (int): Max wait time in seconds for Selenium elements.
            retries (int): Number of retry attempts for downloading PDFs.
            backoff (int): Exponential backoff factor for retries.
        """
        self.wait_time = wait_time
        self.retries = retries
        self.backoff = backoff
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            )
        }

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """Extract and clean text from PDF bytes."""
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text).strip()
        return text

    def download_pdf_with_retries(self, url: str) -> Optional[bytes]:
        """Download a PDF with retry logic and exponential backoff."""
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                wait_time = self.backoff ** attempt
                logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        logging.error(f"All {self.retries} attempts failed for {url}")
        return None

    def get_full_text(self, doi: str) -> Optional[str]:
        """
        Search the UBN repository by DOI and return cleaned PDF text.

        Args:
            doi (str): DOI or search string.

        Returns:
            Optional[str]: Cleaned text from the PDF, or None if not found.
        """
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")  
        options.add_argument("--disable-logging")
        options.add_argument("--log-level=3")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")

        with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
            driver.get("https://repository.ubn.ru.nl/discover")

            try:
                search_input = WebDriverWait(driver, self.wait_time).until(
                    EC.presence_of_element_located((By.ID, "aspect_discovery_SimpleSearch_field_query"))
                )
                search_input.clear()
                search_input.send_keys(doi)
                search_input.submit()

                pdf_link_element = WebDriverWait(driver, self.wait_time).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "image-link"))
                )
                full_text_link = pdf_link_element.get_attribute("href")

            except Exception as e:
                logging.error(f"Error finding elements: {e}")
                return None

        pdf_content = self.download_pdf_with_retries(full_text_link)
        if not pdf_content:
            return None
        logging.info(f"Retrieved full text for: {doi}")
        return self.extract_text_from_pdf(pdf_content)

if __name__ == "__main__":
    scraper = PDFScraper()
    doi = "10.1093/BJS/ZNAE019"
    full_text = scraper.get_full_text(doi)
    print(full_text)