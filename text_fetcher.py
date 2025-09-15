import requests
import time
import re
import os
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from io import BytesIO
from difflib import SequenceMatcher
import subprocess

# --- Dependencies ---
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv


# -------------------- API Keys ---------------------
load_dotenv()
ELSEVIER_KEY = os.getenv("ELSEVIER_KEY")

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
    def fetch_full_paper_text(self, doi: str, title: Optional[str] = None, publisher: Optional[str] = None) -> FullPaperText:
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

                    # validate if methods are in sections
                    methods_found = any(
                        section.section_type == "methods" or
                        re.search(r"\b(methods?|materials\s+and\s+methods?|experimental\s+procedures?)\b",
                                section.section_name, re.I)
                        for section in sections
                    )

                    if methods_found:
                        self.logger.info(f"Successfully fetched and parsed DOI {doi} from PubMed (PMID: {pmid}).")
                        return FullPaperText(
                            doi=doi, pmcid=None, source='PubMed', full_text=full_text,
                            sections=sections, success=True
                        )
                    else:
                        self.logger.warning(f"PMID {pmid} found for DOI {doi}, but no methods section detected. Falling back to UBN/GROBID.")
        
        # Step 3: Try elsevier
        if publisher == "Elsevier BV":
            self.logger.info(f"Attempting to fetch DOI {doi} from Elsevier...")
            url = f'https://api.elsevier.com/content/article/doi/{doi}'
            headers = {'X-ELS-APIKey': ELSEVIER_KEY, 'Accept': 'application/xml'}

            response = requests.get(url, headers=headers)

            xml_text = response.text

            if xml_text:
                full_text, sections = self.extract_sections_from_elsevier(xml_text)
                if sections:
                    return FullPaperText(
                        doi=doi, pmcid=None, source='Elsevier', full_text=full_text,
                        sections=sections, success=True
                    )
                else:
                    self.logger.warning(f"Failed to extract sections from Elsevier XML for DOI {doi}.")
            else:
                self.logger.warning(f"Failed to fetch XML from Elsevier for DOI {doi}.")


        # Step 4: Fall back to UBN/OA
        self.logger.warning(f"Could not find PMCID or PMID for {doi}. Falling back to UBN/GROBID.")
        try:
            # Pass the title from your dataset directly to the validation method
            try:
                pdf_bytes_io = self._get_ubn_pdf_with_validation(doi, target_title=title)
            except Exception as e:
                pdf_bytes_io = self._get_open_acces_pdf(doi)
                if pdf_bytes_io:
                    self.logger.info(f"Successfully downloaded PDF from Open Access for DOI {doi}.")
                else:
                    raise Exception("Failed to download PDF.")

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
            self.logger.error(f"Failed to fetch ubn paper text for {doi}: {e}", exc_info=False)
            return FullPaperText(
                doi=doi, pmcid=None, source='None', full_text="", sections=[],
                success=False, error_message=str(e)
            )
    
    # --- Helper methods ---
    def extract_methods_text(self, sections: List[SectionText]) -> str:
        """
        Extracts the Methods section from GROBID output.
        Falls back to keyword-based extraction if no section is labeled as 'methods'.
        Returns a single string containing all Methods text.
        """
        # Step 1: Try standard methods detection
        methods_texts = []
        for section in sections:
            if section.section_type.lower() == "methods" or \
            re.search(r"\b(methods?|materials\s+and\s+methods?|experimental\s+procedures?)\b",
                        section.section_name, re.I):
                if section.text:
                    methods_texts.append(section.text.strip())

        # If methods were found, return them
        if methods_texts:
            return "\n\n".join(methods_texts)

        # Step 2: Fallback using keyword-based heuristic
        method_keywords = [
            # General
            "animal experiment", "animal study", "in vivo", "preclinical study", 
            "animal model", "ethics statement", "IACUC", "animal protocol",

            # Species
            "mouse", "mice", "rat", "zebrafish", "drosophila", "fruit fly", 
            "xenopus", "c. elegans",

            # Experimental techniques / procedures
            "knockout", "transgenic", "gene editing", "CRISPR", "RNAi", 
            "embryo injection", "tissue collection", "organ collection", 
            "histology", "immunohistochemistry", "western blot", 
            "behavioral assay", "pharmacological treatment",

            # Administration / handling
            "intraperitoneal", "intravenous", "oral gavage", 
            "animal care", "housing conditions", "sacrifice", "euthanasia"
        ]

        fallback_methods = []
        for section in sections:
            if any(kw.lower() in section.section_name.lower() or kw.lower() in section.text.lower()
                for kw in method_keywords):
                fallback_methods.append(section.text.strip())

        return "\n\n".join(fallback_methods)

    def extract_ethics_text(self, sections: List[SectionText]) -> str:
        ethics_texts = []

        # Regex pattern for ethics/committee/animal statements
        pattern = re.compile(
            r"(ethical|ethics|institutional\s+review\s+board(\s+(statement|approval))?|ethics\s+committee|animal\s*(care|use|experiment|study))",
            re.I
        )

        for section in sections:
            if (section.section_name and pattern.search(section.section_name)) \
            or (section.text and pattern.search(section.text)):
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
        return ratio >= threshold

    # --- UBN Scraper and Processor Methods ---
    def _get_ubn_pdf_with_validation(self, doi: str, target_title: Optional[str] = None) -> Optional[BytesIO]:
        """Searches UBN, validates title against the provided target, and returns PDF."""

        pdf_url = None

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

        if target_title and self._is_similar(ubn_title, target_title):
            pdf_link_element = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, "image-link"))
            )
            pdf_url = pdf_link_element.get_attribute("href")

        elif target_title and not self._is_similar(ubn_title, target_title):
            self.logger.info("Titles do not match. Looking for other titles...")
            title_elements = driver.find_elements(By.CSS_SELECTOR, 
                "#aspect_discovery_SimpleSearch_div_search-results div.ds-artifact-item div.artifact-description a h4"
            )
            
            for i, t in enumerate(title_elements):
                if self._is_similar(t.text, target_title):
                    break
            
            if self._is_similar(t.text, target_title):
    
                # Get all artifact items
                artifact_items = driver.find_elements(By.CSS_SELECTOR, 
                    "#aspect_discovery_SimpleSearch_div_search-results div.ds-artifact-item"
                )
                
                # Select the target item using the index
                target_item = artifact_items[i]  
                
                # Locate the <a> with class 'image-link' inside the thumbnail div
                link_element = target_item.find_element(By.CSS_SELECTOR, "div.col-sm-1.hidden-xs div.thumbnail.artifact-preview a.image-link")
                
                # Get the href
                pdf_url = link_element.get_attribute("href")

        if not pdf_url:
            search_input = driver.find_element(By.ID, 'aspect_discovery_SimpleSearch_field_filter_1')
            search_input.clear()
            search_input.send_keys(target_title)  
            search_input.submit() 
            
            try:
                search_result_item = WebDriverWait(driver, self.selenium_wait_time).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.artifact-description")))
                ubn_title = search_result_item.find_element(By.TAG_NAME, "h4").text.strip()
            except Exception:
                raise Exception("Could not find any search result on the UBN page.")

            if self._is_similar(ubn_title, target_title):
                pdf_url = driver.find_element(By.CLASS_NAME, 'image-link').get_attribute("href")


        # Check link validity
        if not 'pdf' in pdf_url.lower():
            self.logger.error(f"Invalid PDF URL: {pdf_url}")
            return None
        
        self.logger.info(f"Found matching PDF at UBN: {pdf_url}")
        pdf_content = self._download_pdf_with_retries(pdf_url)
        return BytesIO(pdf_content) if pdf_content else None

    def _get_open_acces_pdf(self, doi: str) -> Optional[BytesIO]:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        data = requests.get(url).json()
        link = data.get('open_access').get('oa_url')
        if not link:
            return None
        else:
            pdf_content = self._download_pdf_with_retries(link)
            return BytesIO(pdf_content) if pdf_content else None


    def _download_pdf_with_retries(self, url: str) -> Optional[bytes]:
        for attempt in range(1, self.download_retries + 1):
            try:
                response = requests.get(url, headers=self.scraper_headers, timeout=20)

                # Immediately shutdown on 403
                if response.status_code == 403:
                    self.logger.error(f"403 Forbidden encountered for {url}. Shutting down.")
                    return None

                response.raise_for_status()
                return response.content

            except requests.RequestException as e:
                wait = self.download_backoff ** attempt
                self.logger.warning(f"PDF download attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        self.logger.error(f"All {self.download_retries} download attempts failed for {url}")
        return None

    def _is_grobid_running(self) -> bool:
        try:
            resp = requests.get("http://localhost:8070/api/isalive", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
    

    def _start_grobid_docker(self):
        self.logger.info("Starting GROBID Docker container...")
        # Run in detached mode
        subprocess.Popen([
            "docker", "run", "--rm", "--init", "-p", "8070:8070", "grobid/grobid:0.8.2-full"
        ])
        
        # Wait until GROBID is ready
        for _ in range(30):  # ~30 seconds max
            if self._is_grobid_running():
                self.logger.info("GROBID is up and running!")
                return
            time.sleep(1)
        
        raise RuntimeError("GROBID failed to start within 30 seconds.")


    def _send_pdf_to_grobid(self, pdf_file: BytesIO) -> Optional[str]:
        url = "http://localhost:8070/api/processFulltextDocument"

        # Ensure GROBID is running
        if not self._is_grobid_running():
            self._start_grobid_docker()

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
    
    def extract_sections_from_elsevier(self, xml_text: str) -> tuple[str, List[SectionText]]:
        sections: List[SectionText] = []
        full_text = ""

        try:
            soup = BeautifulSoup(xml_text, "lxml-xml")

            # Full text
            full_text_elem = soup.find("ce:doc") or soup.find("xocs:doc")
            full_text = full_text_elem.get_text(" ", strip=True) if full_text_elem else ""

            # Parse main sections in body, front, back
            for tag_name in ['front', 'body', 'back']:
                tag = soup.find(tag_name)
                if tag:
                    for sec in tag.find_all("ce:section"):
                        # Section title
                        title_elem = sec.find("ce:section-title")
                        title_text = title_elem.get_text(" ", strip=True) if title_elem else f"section_{len(sections)}"

                        # Determine section type based on title
                        title_lower = title_text.lower()
                        section_type = "body"
                        if "method" in title_lower or "material" in title_lower:
                            section_type = "methods"
                        elif "result" in title_lower or "finding" in title_lower:
                            section_type = "results"
                        elif "discussion" in title_lower or "conclusion" in title_lower:
                            section_type = "discussion"
                        elif "introduction" in title_lower or "background" in title_lower:
                            section_type = "introduction"

                        # Section text
                        sec_text = sec.get_text(" ", strip=True)
                        if sec_text:
                            sections.append(SectionText(
                                section_name=title_text,
                                text=sec_text,
                                section_type=section_type
                            ))

        except Exception as e:
            sections = []
            full_text = ""

        return full_text, sections



if __name__ == "__main__":
    fetcher = PaperFetcher()
    paper = fetcher.fetch_full_paper_text("10.1242/JCS.218487", "Sulfur in lucinid bivalves inhibits intake rates of a molluscivore shorebird", '')
    methods_text = fetcher.extract_methods_text(paper.sections)
    print(methods_text)
    #10.1890/14-0082.1
    #10.1016/J.BEPROC.2015.08.013