import requests
import time
import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class SectionText:
    """Represents text from a specific section of a paper"""
    section_name: str
    text: str
    section_type: str  # 'front', 'body', 'back', 'methods', 'results', etc.

@dataclass
class FullPaperText:
    """Represents the full paper text with sections"""
    doi: str
    pmcid: Optional[str]
    full_text: str
    sections: List[SectionText]
    success: bool
    error_message: str = ""

class PMCTextFetcher:
    """
    Fetches full paper text from PubMed Central
    Uses the working approach from insp file
    """
    
    def __init__(self, tool_name: str = "animal_study_classifier", email: str = "research@example.com"):
        self.ncbi_tool = tool_name
        self.ncbi_email = email
        self.ncbi_api_key = None
        self.polite_delay = 0.34
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def http_get(self, url: str, params: Dict[str, str], expect_json: bool = False) -> requests.Response:
        """
        Wrapper around requests.get with retries and basic error handling.
        Appends NCBI tool/email/key parameters automatically.
        """
        params = dict(params or {})
        params.setdefault("tool", self.ncbi_tool)
        params.setdefault("email", self.ncbi_email)
        if self.ncbi_api_key:
            params.setdefault("api_key", self.ncbi_api_key)

        tries = 3
        for attempt in range(1, tries + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    if expect_json:
                        # Validate JSON when we say we expect JSON
                        _ = resp.json()
                    return resp
                # Simple backoff
                time.sleep(0.5 * attempt)
            except Exception as e:
                self.logger.warning(f"Error GET {url}: {e} (attempt {attempt})")
                time.sleep(0.5 * attempt)
        
        resp.raise_for_status()
        return resp
    
    def normalize_identifier(self, id_str: str) -> str:
        """
        Accept either a full DOI or a Springer/BMC suffix like 's40478-023-01698-4'.
        """
        id_str = id_str.strip()
        if id_str.lower().startswith("doi:"):
            id_str = id_str[4:].strip()

        if id_str.startswith("10."):
            return id_str

        # Heuristic for Springer/BMC-style suffix
        if re.fullmatch(r"s\d{5}-\d{3}-\d{5}-\d", id_str):
            # Try BMC first
            return "10.1186/" + id_str

        return id_str
    
    def doi_to_pmcid_via_idconv(self, doi: str) -> Optional[str]:
        """
        Use PMC's ID Converter to map DOI -> PMCID.
        """
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {"format": "json", "ids": doi}
        try:
            resp = self.http_get(url, params, expect_json=True)
            data = resp.json()
            if "records" in data and data["records"]:
                rec = data["records"][0]
                pmcid = rec.get("pmcid")
                return pmcid
        except Exception as e:
            self.logger.warning(f"[idconv] WARN for {doi}: {e}")
        return None
    
    def doi_to_pmcid_via_esearch(self, doi: str) -> Optional[str]:
        """
        Fallback: search PMC for the DOI.
        """
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"{doi}[DOI]",
            "retmode": "json",
            "retmax": "1",
        }
        try:
            resp = self.http_get(url, params, expect_json=True)
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if ids:
                return "PMC" + ids[0]
        except Exception as e:
            self.logger.warning(f"[esearch] WARN for {doi}: {e}")
        return None
    
    def doi_to_pmcid(self, doi_or_suffix: str) -> Optional[str]:
        """
        Resolve whatever the user provided to a PMCID, if possible.
        """
        doi = self.normalize_identifier(doi_or_suffix)

        pmcid = self.doi_to_pmcid_via_idconv(doi)
        if pmcid:
            return pmcid
        time.sleep(self.polite_delay)

        pmcid = self.doi_to_pmcid_via_esearch(doi)
        if pmcid:
            return pmcid

        return None
    
    def fetch_pmc_jats_xml(self, pmcid: str) -> Optional[str]:
        """
        Use E-utilities efetch to retrieve full text JATS XML for a PMCID.
        """
        # efetch for PMC expects the numeric id (without 'PMC') OR with PMC — both generally work.
        numeric = pmcid.replace("PMC", "")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": numeric,
            "retmode": "xml",
        }
        try:
            resp = self.http_get(url, params, expect_json=False)
            if resp.text.strip():
                return resp.text
        except Exception as e:
            self.logger.warning(f"[efetch] WARN for {pmcid}: {e}")
        return None
    
    def extract_sections_from_jats(self, xml_text: str) -> List[SectionText]:
        """Extract text from different sections of JATS XML format"""
        sections = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xml_text, "lxml-xml")
            
            # Extract front matter (title, abstract, affiliations, keywords, etc.)
            front = soup.find("front")
            if front:
                front_text = front.get_text(separator=" ", strip=True)
                if front_text:
                    sections.append(SectionText("front", front_text, "front"))
            
            # Extract body text with section identification
            body = soup.find("body")
            if body:
                # Find all sections in the body
                for sec in body.find_all("sec"):
                    title_elem = sec.find("title")
                    title_text = title_elem.get_text(" ", strip=True) if title_elem else ""
                    
                    # Determine section type based on title
                    section_type = "body"
                    if re.search(r"\b(methods?|materials?\s+and\s+methods?|experimental\s+procedures?)\b", title_text, re.I):
                        section_type = "methods"
                    elif re.search(r"\b(results?|findings?)\b", title_text, re.I):
                        section_type = "results"
                    elif re.search(r"\b(discussion|conclusion)\b", title_text, re.I):
                        section_type = "discussion"
                    elif re.search(r"\b(introduction|background)\b", title_text, re.I):
                        section_type = "introduction"
                    
                    # Get section text
                    section_text = sec.get_text(separator=" ", strip=True)
                    if section_text:
                        section_name = title_text if title_text else f"body_section_{len(sections)}"
                        sections.append(SectionText(section_name, section_text, section_type))
                
                # If no sections found, get all body text
                if not sections:
                    # Skip references to avoid noise
                    for ref_list in body.find_all("ref-list"):
                        ref_list.decompose()
                    body_text = body.get_text(separator=" ", strip=True)
                    if body_text:
                        sections.append(SectionText("body", body_text, "body"))
            
            # Extract back matter (sometimes contains ethics statements)
            back = soup.find("back")
            if back:
                back_text = back.get_text(separator=" ", strip=True)
                if back_text:
                    sections.append(SectionText("back", back_text, "back"))
                    
        except Exception as e:
            self.logger.warning(f"JATS parsing failed: {e}")
        
        return sections
    
    def extract_full_text_from_sections(self, sections: List[SectionText]) -> str:
        """Combine all sections into full text"""
        texts = []
        for section in sections:
            if section.text:
                texts.append(section.text)
        return " ".join(texts)
    
    def extract_methods_text(self, sections: List[SectionText]) -> str:
        """Return the concatenated text of all Methods-like sections."""
        if not sections:
            return ""
        methods_texts: List[str] = []
        for section in sections:
            try:
                section_type = (section.section_type or "").lower()
                section_name = (section.section_name or "").lower()
                if section_type == "methods" or re.search(r"\b(methods?|materials\s+and\s+methods?|experimental\s+procedures?)\b", section_name, re.I):
                    if section.text:
                        methods_texts.append(section.text)
            except Exception:
                continue
        return " \n\n".join(methods_texts).strip()
    
    def fetch_methods_text(self, doi: str) -> str:
        """Fetch full text for DOI and return only the Methods section text if available."""
        paper = self.fetch_full_paper_text(doi)
        if not paper.success:
            self.logger.warning(f"Failed to fetch full paper text for {doi}: {paper.error_message}")
            return ""
        return self.extract_methods_text(paper.sections)

    def extract_ethics_text(self, sections: List[SectionText]) -> str:
        """Return the concatenated text of all Ethics-like sections."""
        if not sections:
            return ""
        ethics_texts: List[str] = []
        for section in sections:
            try:
                section_type = (section.section_type or "").lower()
                section_name = (section.section_name or "").lower()
                if section_type == "ethical approval" or re.search(r"\b(ethical\s+approval|animal\s+care|animal\s+use|animal\s+experiment|animal\s+study)\b", section_name, re.I):
                    if section.text:
                        ethics_texts.append(section.text)
            except Exception:
                continue
        return " \n\n".join(ethics_texts).strip()
    
    def fetch_full_paper_text(self, doi: str) -> FullPaperText:
        """
        Fetch full paper text from PMC and return structured data with sections.
        """
        try:
            # Convert DOI to PMCID
            pmcid = self.doi_to_pmcid(doi)
            if not pmcid:
                return FullPaperText(
                    doi=doi,
                    pmcid=None,
                    full_text="",
                    sections=[],
                    success=False,
                    error_message="Failed to convert DOI to PMCID"
                )
            
            # Fetch JATS XML
            xml_text = self.fetch_pmc_jats_xml(pmcid)
            if not xml_text:
                return FullPaperText(
                    doi=doi,
                    pmcid=pmcid,
                    full_text="",
                    sections=[],
                    success=False,
                    error_message="Failed to fetch JATS XML"
                )
            
            # Extract sections
            sections = self.extract_sections_from_jats(xml_text)
            if not sections:
                return FullPaperText(
                    doi=doi,
                    pmcid=pmcid,
                    full_text="",
                    sections=[],
                    success=False,
                    error_message="Failed to extract text from JATS XML"
                )
            
            # Combine into full text
            full_text = self.extract_full_text_from_sections(sections)
            
            #self.logger.info(f"Successfully fetched full paper text for {doi} ({len(full_text)} chars, {len(sections)} sections)")
            
            return FullPaperText(
                doi=doi,
                pmcid=pmcid,
                full_text=full_text,
                sections=sections,
                success=True
            )
            
        except Exception as e:
            self.logger.warning(f"Full paper text fetch failed for {doi}: {e}")
            return FullPaperText(
                doi=doi,
                pmcid=None,
                full_text="",
                sections=[],
                success=False,
                error_message=str(e)
            )


if __name__ == "__main__":
    fetcher = PMCTextFetcher()
    paper = fetcher.fetch_full_paper_text('10.1016/J.CTRO.2024.100875')
    method = fetcher.extract_ethics_text(paper.sections)
    print(method)