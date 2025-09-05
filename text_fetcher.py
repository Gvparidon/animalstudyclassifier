import requests
import time
import re
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from io import BytesIO
from difflib import SequenceMatcher

# --- Dependencies ---
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataclasses ---
@dataclass
class SectionText:
    """Represents text from a specific section of a paper"""
    section_name: str
    text: str
    section_type: str

@dataclass
class FullPaperText:
    """Represents the full paper text with sections"""
    doi: str
    pmcid: Optional[str]
    source: str
    full_text: str
    sections: List[SectionText]
    success: bool
    error_message: str = ""

# --- Combined Fetcher Class ---

class PaperFetcher:
    """
    Fetches full paper text from scientific publications using a tiered approach.
    1. Tries to fetch structured JATS XML from PubMed Central (PMC).
    2. If a PMCID is not found, it falls back to scraping the UBN repository,
       validating the paper's title (if provided), downloading the PDF, and
       processing it with a local GROBID service.
    """

    def __init__(
        self,
        tool_name: str = "animal_study_classifier",
        email: str = "research@example.com",
        selenium_wait_time: int = 10
    ):
        self.ncbi_tool = tool_name
        self.ncbi_email = email
        self.ncbi_api_key = None
        self.polite_delay = 0.34
        self.selenium_wait_time = selenium_wait_time
        self.download_retries = 3
        self.download_backoff = 2
        self.scraper_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        self._driver = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_driver(self):
        if self._driver is None:
            self.logger.info("Initializing Selenium WebDriver for UBN fallback...")
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("--headless=new")
                options.add_argument("--disable-logging")
                options.add_argument("--log-level=3")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-dev-shm-usage")
                self._driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                self._driver.get("https://repository.ubn.ru.nl/discover")
            except Exception as e:
                self.logger.error(f"Failed to initialize Selenium WebDriver: {e}")
                raise
        return self._driver

    def close(self):
        if self._driver:
            self.logger.info("Closing Selenium WebDriver.")
            self._driver.quit()
            self._driver = None

    # --- Core Fetching Logic ---
    def fetch_full_paper_text(self, doi: str, title: Optional[str] = None) -> FullPaperText:
        """
        Main method to fetch paper text. Tries PMC first, then PMID, then falls back to UBN/GROBID.

        Args:
            doi (str): The DOI of the paper.
            title (Optional[str]): The title of the paper, used for validation during UBN fallback.
        """
        # Step 1: Try PMC (full text)
        self.logger.info(f"Attempting to fetch DOI {doi} from PMC...")
        pmcid = self._doi_to_pmcid(doi)

        if pmcid:
            xml_text = self._fetch_pmc_jats_xml(pmcid)
            if xml_text:
                sections = self._extract_sections_from_jats(xml_text)
                if sections:
                    full_text = self._extract_full_text_from_sections(sections)
                    self.logger.info(f"Successfully fetched and parsed DOI {doi} from PMC (PMCID: {pmcid}).")
                    return FullPaperText(
                        doi=doi, pmcid=pmcid, source='PMC', full_text=full_text,
                        sections=sections, success=True
                    )

        # Step 2: Try PubMed (abstract and metadata)
        self.logger.info(f"PMC not available for {doi}. Trying PubMed...")
        pmid = self._doi_to_pmid(doi)
        
        if pmid:
            xml_text = self._fetch_pubmed_xml(pmid)
            if xml_text:
                sections = self._extract_sections_from_pubmed_xml(xml_text)
                if sections:
                    full_text = self._extract_full_text_from_sections(sections)
                    self.logger.info(f"Successfully fetched and parsed DOI {doi} from PubMed (PMID: {pmid}).")
                    return FullPaperText(
                        doi=doi, pmcid=None, source='PubMed', full_text=full_text,
                        sections=sections, success=True
                    )

        # Step 3: Fall back to UBN/GROBID
        self.logger.warning(f"Could not find PMCID or PMID for {doi}. Falling back to UBN/GROBID.")
        try:
            # Pass the title from your dataset directly to the validation method
            pdf_bytes_io = self._get_ubn_pdf_with_validation(doi, target_title=title)
            if not pdf_bytes_io:
                raise Exception("Failed to download PDF from UBN.")

            tei_xml = self._send_pdf_to_grobid(pdf_bytes_io)
            if not tei_xml:
                raise Exception("Failed to process PDF with GROBID.")
            
            sections = self._extract_sections_from_tei_xml(tei_xml)
            if not sections:
                raise Exception("Failed to extract sections from GROBID's TEI XML.")

            full_text = self._extract_full_text_from_sections(sections)
            self.logger.info(f"Successfully fetched and parsed DOI {doi} from UBN/GROBID.")
            return FullPaperText(
                doi=doi, pmcid=None, source='UBN/GROBID', full_text=full_text,
                sections=sections, success=True
            )
        except Exception as e:
            error_message = f"All methods failed for {doi}: {e}"
            self.logger.error(error_message, exc_info=False)
            return FullPaperText(
                doi=doi, pmcid=None, source='None', full_text="", sections=[],
                success=False, error_message=str(e)
            )
    
    # --- Helper methods ---
    def extract_methods_text(self, sections: List[SectionText]) -> str:
        methods_texts = []
        for section in sections:
            if section.section_type == "methods" or re.search(r"\b(methods?|materials\s+and\s+methods?|experimental\s+procedures?)\b", section.section_name, re.I):
                if section.text:
                    methods_texts.append(section.text)
        return " \n\n".join(methods_texts).strip()

    def extract_ethics_text(self, sections: List[SectionText]) -> str:
        ethics_texts = []
        for section in sections:
            if re.search(r"\b(ethical\s+approval|animal\s+care|animal\s+use|animal\s+experiment|animal\s+study)\b", section.section_name, re.I):
                if section.text:
                    ethics_texts.append(section.text)
        return " \n\n".join(ethics_texts).strip()

    # --- Internal Methods for PMC  ---
    def _http_get(self, url, params, expect_json=False):
        params = dict(params or {})
        params.setdefault("tool", self.ncbi_tool)
        params.setdefault("email", self.ncbi_email)
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    if expect_json: _ = resp.json()
                    return resp
                time.sleep(0.5 * attempt)
            except Exception as e:
                self.logger.warning(f"Error GET {url}: {e} (attempt {attempt})")
                time.sleep(0.5 * attempt)
        resp.raise_for_status()

    def _normalize_identifier(self, id_str):
        id_str = id_str.strip()
        if id_str.lower().startswith("doi:"): id_str = id_str[4:].strip()
        if id_str.startswith("10."): return id_str
        if re.fullmatch(r"s\d{5}-\d{3}-\d{5}-\d", id_str): return "10.1186/" + id_str
        return id_str

    def _doi_to_pmcid(self, doi_or_suffix):
        doi = self._normalize_identifier(doi_or_suffix)
        try:
            resp = self._http_get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/", {"format": "json", "ids": doi}, expect_json=True)
            rec = resp.json().get("records", [{}])[0]
            if rec.get("pmcid"): return rec["pmcid"]
        except Exception as e: self.logger.warning(f"[idconv] WARN for {doi}: {e}")
        time.sleep(self.polite_delay)
        try:
            resp = self._http_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "pmc", "term": f"{doi}[DOI]", "retmode": "json", "retmax": "1"}, expect_json=True)
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if ids: return "PMC" + ids[0]
        except Exception as e: self.logger.warning(f"[esearch] WARN for {doi}: {e}")
        return None

    def _doi_to_pmid(self, doi_or_suffix):
        """Convert DOI to PMID using NCBI ID Converter and ESearch"""
        doi = self._normalize_identifier(doi_or_suffix)
        try:
            resp = self._http_get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/", {"format": "json", "ids": doi}, expect_json=True)
            rec = resp.json().get("records", [{}])[0]
            if rec.get("pmid"): return rec["pmid"]
        except Exception as e: self.logger.warning(f"[idconv] PMID WARN for {doi}: {e}")
        time.sleep(self.polite_delay)
        try:
            resp = self._http_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "pubmed", "term": f"{doi}[DOI]", "retmode": "json", "retmax": "1"}, expect_json=True)
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if ids: return ids[0]
        except Exception as e: self.logger.warning(f"[esearch] PMID WARN for {doi}: {e}")
        return None

    def _fetch_pmc_jats_xml(self, pmcid):
        numeric = pmcid.replace("PMC", "")
        try:
            resp = self._http_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", {"db": "pmc", "id": numeric, "retmode": "xml"})
            return resp.text.strip()
        except Exception as e: self.logger.warning(f"[efetch] WARN for {pmcid}: {e}")
        return None

    def _fetch_pubmed_xml(self, pmid):
        """Fetch PubMed XML using PMID (contains abstract and metadata)"""
        try:
            resp = self._http_get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", {"db": "pubmed", "id": pmid, "retmode": "xml"})
            return resp.text.strip()
        except Exception as e: self.logger.warning(f"[efetch] PMID WARN for {pmid}: {e}")
        return None

    def _extract_sections_from_jats(self, xml_text):
        sections = []
        try:
            soup = BeautifulSoup(xml_text, "lxml-xml")
            for tag_name in ['front', 'body', 'back']:
                tag = soup.find(tag_name)
                if tag:
                    for sec in tag.find_all("sec"):
                        title_elem = sec.find("title")
                        title_text = title_elem.get_text(" ", strip=True) if title_elem else f"section_{len(sections)}"
                        section_type="body"; title_lower=title_text.lower()
                        if "method" in title_lower or "material" in title_lower: section_type="methods"
                        elif "result" in title_lower or "finding" in title_lower: section_type="results"
                        elif "discussion" in title_lower or "conclusion" in title_lower: section_type="discussion"
                        elif "introduction" in title_lower or "background" in title_lower: section_type="introduction"
                        sec_text = sec.get_text(separator=" ", strip=True)
                        if sec_text: sections.append(SectionText(title_text, sec_text, section_type))
        except Exception as e: self.logger.error(f"JATS XML parsing failed: {e}")
        return sections

    def _extract_sections_from_pubmed_xml(self, xml_text):
        """Extract sections from PubMed XML (mainly abstract and metadata)"""
        sections = []
        try:
            soup = BeautifulSoup(xml_text, "lxml-xml")
            
            # Extract title
            title_elem = soup.find("ArticleTitle")
            if title_elem:
                title_text = title_elem.get_text(strip=True)
                if title_text:
                    sections.append(SectionText("Title", title_text, "title"))
            
            # Extract abstract
            abstract_elem = soup.find("Abstract")
            if abstract_elem:
                # Handle structured abstracts
                abstract_texts = []
                for abstract_text in abstract_elem.find_all("AbstractText"):
                    label = abstract_text.get("Label", "")
                    text = abstract_text.get_text(strip=True)
                    if text:
                        if label:
                            abstract_texts.append(f"{label}: {text}")
                        else:
                            abstract_texts.append(text)
                
                if abstract_texts:
                    full_abstract = " ".join(abstract_texts)
                    sections.append(SectionText("Abstract", full_abstract, "abstract"))
            
            # Extract keywords/MeSH terms
            mesh_headings = soup.find_all("MeshHeading")
            if mesh_headings:
                mesh_terms = []
                for mesh in mesh_headings:
                    descriptor = mesh.find("DescriptorName")
                    if descriptor:
                        mesh_terms.append(descriptor.get_text(strip=True))
                if mesh_terms:
                    mesh_text = "; ".join(mesh_terms)
                    sections.append(SectionText("MeSH Terms", mesh_text, "keywords"))
            
        except Exception as e: 
            self.logger.error(f"PubMed XML parsing failed: {e}")
        return sections
    
    def _extract_full_text_from_sections(self, sections):
        return " ".join([section.text for section in sections if section.text])

    # --- Title Similarity Helper ---
    def _is_similar(self, a: str, b: str, threshold: float = 0.8) -> bool:
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        self.logger.info(f"Comparing titles. Similarity score: {ratio:.2f} (Threshold: {threshold})")
        return ratio >= threshold

    # --- UBN Scraper and Processor Methods ---
    def _get_ubn_pdf_with_validation(self, doi: str, target_title: Optional[str] = None) -> Optional[BytesIO]:
        """Searches UBN, validates title against the provided target, and returns PDF."""
        if not target_title:
            self.logger.warning("No target title provided. Proceeding with UBN download without validation.")

        driver = self._get_driver()
        driver.get("https://repository.ubn.ru.nl/discover")
        
        search_input = WebDriverWait(driver, self.selenium_wait_time).until(EC.presence_of_element_located((By.ID, "aspect_discovery_SimpleSearch_field_query")))
        search_input.clear()
        search_input.send_keys(doi)
        search_input.submit()

        try:
            search_result_item = WebDriverWait(driver, self.selenium_wait_time).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.artifact-description")))
            ubn_title = search_result_item.find_element(By.TAG_NAME, "h4").text.strip()
        except Exception:
            raise Exception("Could not find any search result on the UBN page.")

        if target_title:
            self.logger.info(f"Target title: '{target_title}'")
            self.logger.info(f"UBN found:    '{ubn_title}'")
            if not self._is_similar(ubn_title, target_title):
                raise Exception(f"Title mismatch on UBN (score < 80%). Aborting download.")
            self.logger.info("Title match successful.")

        pdf_link_element = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "image-link"))
        )
        pdf_url = pdf_link_element.get_attribute("href")
        
        self.logger.info(f"Found matching PDF at UBN: {pdf_url}")
        pdf_content = self._download_pdf_with_retries(pdf_url)
        return BytesIO(pdf_content) if pdf_content else None
        
    def _download_pdf_with_retries(self, url: str) -> Optional[bytes]:
        for attempt in range(1, self.download_retries + 1):
            try:
                response = requests.get(url, headers=self.scraper_headers, timeout=20)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                wait = self.download_backoff ** attempt
                self.logger.warning(f"PDF download attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        self.logger.error(f"All {self.download_retries} download attempts failed for {url}")
        return None

    def _send_pdf_to_grobid(self, pdf_file: BytesIO) -> Optional[str]:
        url = "http://localhost:8070/api/processFulltextDocument"
        self.logger.info("Sending PDF to GROBID for processing...")
        try:
            pdf_file.seek(0)
            files = {"input": pdf_file}
            resp = requests.post(url, files=files, timeout=120)
            resp.raise_for_status()
            self.logger.info("Successfully received response from GROBID.")
            return resp.text
        except requests.RequestException as e: self.logger.error(f"Error communicating with GROBID: {e}")
        return None

    def _extract_sections_from_tei_xml(self, tei_xml: str) -> List[SectionText]:
        self.logger.info("Parsing TEI XML from GROBID...")
        sections = []
        try:
            tree = etree.fromstring(tei_xml.encode("utf-8"))
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            divs = tree.xpath("//tei:body//tei:div[tei:head]", namespaces=ns)
            for div in divs:
                head_elem = div.find("tei:head", namespaces=ns)
                if head_elem is None or head_elem.text is None: continue
                section_name = head_elem.text.strip()
                p_texts = ["".join(p.itertext()).strip() for p in div.findall(".//tei:p", namespaces=ns)]
                text = "\n\n".join(filter(None, p_texts))
                if not text: continue
                section_type="body"; name_lower = section_name.lower()
                if "method" in name_lower or "material" in name_lower: section_type="methods"
                elif "result" in name_lower or "finding" in name_lower: section_type="results"
                elif "discussion" in name_lower or "conclusion" in name_lower: section_type="discussion"
                elif "introduction" in name_lower or "background" in name_lower: section_type="introduction"
                sections.append(SectionText(section_name, text, section_type))
        except Exception as e: self.logger.error(f"TEI XML parsing failed: {e}")
        return sections
