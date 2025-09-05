import re
import time
import logging
from io import BytesIO
from typing import Optional
from lxml import etree  

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class UBNTextFetcher:
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
        self.driver = self._init_driver()  

    def _init_driver(self):
        """Initialize and return a Selenium Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-logging")
        options.add_argument("--log-level=3")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://repository.ubn.ru.nl/discover")  # Load page once
        return driver

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

    def get_pdf(self, doi: str) -> Optional[str]:
        """Search the UBN repository by DOI and return PDF."""
        driver = self.driver
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
        return BytesIO(pdf_content)

    def send_pdf_to_gorbid(self, pdf_file: BytesIO) -> Optional[str]:
        """Send PDF content to Gorbid and return xml response."""

        # GROBID endpoint
        url = "http://localhost:8070/api/processFulltextDocument"

        try:
            # Send PDF to GROBID
            pdf_file.seek(0)  # Ensure we're at the start of the file
            resp = requests.post(url, files={"input": pdf_file})
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logging.error(f"Error sending PDF to Gorbid: {e}")
            return None

    def extract_methods_from_xml(self, tei_xml: str) -> Optional[str]:
        """Extract methods from xml response."""

        # Parse TEI XML
        tree = etree.fromstring(tei_xml.encode("utf-8"))
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        methods_text = []

        # 1. Find the <div> with <head>Methods</head>
        methods_divs = tree.xpath(
            "//tei:div[tei:head[contains(translate(text(), 'METHOD', 'method'), 'method')]]",
            namespaces=ns
        )

        if methods_divs:
            # 2. Iterate over following <div> siblings to gather the Methods content
            div = methods_divs[0]
            for sibling in div.itersiblings():
                # Stop if we reach a sibling that is likely a new top-level section
                head = sibling.find("tei:head", namespaces=ns)
                if head is not None and any(word in head.text.lower() for word in ["result", "discussion", "conclusion"]):
                    break
                # Collect paragraphs
                paragraphs = sibling.xpath(".//tei:p//text()", namespaces=ns)
                methods_text.extend(paragraphs)

        methods_text = "\n\n".join([p.strip() for p in methods_text if p.strip()])

        if not methods_text:
            return None
        else:
            return methods_text
    
    def extract_methods_for_doi(self, doi: str) -> Optional[str]:
        """Extract methods for a specific DOI."""
        pdf_file = self.get_pdf(doi)
        if not pdf_file:
            return None
        tei_xml = self.send_pdf_to_gorbid(pdf_file)
        if not tei_xml:
            return None
        return self.extract_methods_from_xml(tei_xml)

    def close(self):
        """Close the Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None

if __name__ == "__main__":
    import time
    start_time = time.time()
    scraper = UBNTextFetcher()
    dois = [
    "10.1016/J.CUB.2024.08.048"
    ]

    for doi in dois:
        methods = scraper.extract_methods_for_doi(doi)
        if methods:
            print(methods)

    scraper.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")