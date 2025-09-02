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


    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """Extract and clean text from PDF bytes."""
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            text = text.replace("\xa0", " ")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n+", "\n", text).strip()
            return text
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}")
            return ""

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
        """Search the UBN repository by DOI and return cleaned PDF text."""
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
        return self.extract_text_from_pdf(pdf_content)

    def close(self):
        """Close the Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None

if __name__ == "__main__":
    import time
    start_time = time.time()
    scraper = PDFScraper()
    dois = dois = [
    "10.1093/JAC/DKAE232",
    "10.1186/S40359-024-01904-5",
    "10.1080/10428194.2024.2396542",
    "10.1016/J.BPJ.2023.11.3399",
    "10.1080/07853890.2024.2399755",
    "10.1097/QAD.0000000000003852",
    "10.1038/S41380-023-02022-1",
    "10.1016/S1473-3099(23)00495-4",
    "10.1016/J.JAD.2024.01.049",
    "10.4049/JIMMUNOL.2400065",
    "10.1038/S41375-024-02145-6",
    "10.1080/03009742.2023.2213509",
    "10.1111/HEX.70105",
    "10.1097/YIC.0000000000000531",
    "10.1136/IJGC-2023-004781",
    "10.1007/S00429-024-02773-9",
    "10.1016/J.BBI.2024.05.037",
    "10.1177/17474930241242625",
    "10.1016/J.ADRO.2023.101379",
    "10.1002/AJMG.A.63472",
    "10.1016/J.EUROS.2023.11.004",
    "10.1136/BMJOPEN-2024-092165",
    "10.1002/HBM.70025",
    "10.1016/S2666-7568(24)00068-0",
    "10.1016/J.JACC.2024.03.396",
    "10.3390/JPM14050523",
    "10.1111/HEX.13949",
    "10.1016/J.DENTAL.2024.09.002",
    "10.1093/AGEING/AFAE248",
    "10.4244/EIJ-D-24-00089",
    "10.1186/S12245-024-00799-8",
    "10.1186/S13195-024-01609-2",
    "10.1016/J.MEX.2023.102543",
    "10.1016/J.NMD.2024.03.001",
    "10.1212/NXG.0000000000200214",
    "10.1177/1877718X241296016",
    "10.1111/DOM.15649",
    "10.1017/S0963180122000718",
    "10.1111/1471-0528.17591",
    "10.1016/J.NEUROIMAGE.2024.120849",
    "10.1111/APHA.14150",
    "10.1002/UEG2.12661",
    "10.1016/J.YMGME.2024.108144",
    "10.1136/BJSPORTS-2023-107190",
    "10.1016/J.JAD.2024.06.093",
    "10.1080/13651501.2024.2409654",
    "10.1093/CKJ/SFAE122",
    "10.1001/JAMANETWORKOPEN.2024.39571",
    "10.1016/J.XCRM.2024.101529"
    ]

    for doi in dois:
        print(doi)
        full_text = scraper.get_full_text(doi)
    scraper.close()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")