import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from pmc_text_fetcher import PMCTextFetcher, FullPaperText, SectionText

@dataclass
class EthicsEvidence:
    """Structured evidence for ethics extraction"""
    category: str
    keywords_found: List[str]
    sentences: List[str]
    institutions: List[str]
    section_sources: List[str]  # Which sections the evidence came from

class EthicsExtractor:
    """
    Ethics extraction system
    Uses full paper text from PMC for better analysis
    """
    
    def __init__(self, tool_name: str = "ethics_extractor", email: str = "hi.hoi@mail.nl"):
        # Initialize PMC text fetcher
        self.pmc_fetcher = PMCTextFetcher(tool_name=tool_name, email=email)
        
        # Ethics patterns - core concepts only
        self.ethics_patterns = {
            # Committees and boards
            "ethics_committee": re.compile(r"\b(ethics\s+committee|institutional\s+review\s+board|IRB|ethical\s+committee|research\s+ethics\s+committee|REC|animal\s+ethics\s+committee|AEC)\b", re.I),
            "animal_care": re.compile(r"\b(animal\s+care\s+and\s+use\s+committee|ACUC|IACUC|institutional\s+animal\s+care\s+and\s+use\s+committee|animal\s+welfare\s+committee|AWC)\b", re.I),
            "institutional_approval": re.compile(r"\b(institutional\s+approval|university\s+approval|faculty\s+approval|department\s+approval|institute\s+approval|center\s+approval)\b", re.I),
            
            # Guidelines and compliance
            "guidelines": re.compile(r"\b(NIH\s+Guide|ARRIVE|3R|replacement|reduction|refinement|animal\s+welfare|welfare\s+guidelines|ethical\s+guidelines|research\s+guidelines)\b", re.I),
            "regulations": re.compile(r"\b(regulations|regulatory\s+compliance|compliance|ethical\s+standards|standards|protocol|ethical\s+protocol|animal\s+protocol)\b", re.I),
            
            # Approval and consent
            "approval": re.compile(r"\b(approved|approval|permission|authorized|authorization|consent|informed\s+consent|ethical\s+approval|ethical\s+permission)\b", re.I),
            
            # Declaration statements
            "declaration": re.compile(r"\b(conducted\s+in\s+accordance|in\s+accordance\s+with|following\s+the|according\s+to|compliant\s+with|adhering\s+to|following\s+guidelines|ethical\s+principles)\b", re.I),
            "ethical_conduct": re.compile(r"\b(ethical\s+conduct|ethical\s+standards|ethical\s+principles|ethical\s+practices|ethical\s+considerations|ethical\s+requirements)\b", re.I),
            
            # Animal welfare
            "animal_welfare": re.compile(r"\b(animal\s+welfare|welfare\s+of\s+animals|humane\s+treatment|humane\s+care|animal\s+rights|animal\s+protection|welfare\s+considerations)\b", re.I),
            "minimize_suffering": re.compile(r"\b(minimize\s+suffering|reduce\s+pain|alleviate\s+distress|prevent\s+suffering|minimize\s+stress|reduce\s+discomfort|humane\s+endpoints)\b", re.I),
            
            # Institutions
            "institution_mention": re.compile(r"\b(university|institute|center|faculty|department|school|college|hospital|medical\s+center|research\s+center|laboratory|lab)\b", re.I),
        }
        
        # Context patterns for sentence extraction
        self.context_patterns = {
            "ethics_section": re.compile(r"\b(ethics|ethical|welfare|approval|committee|guidelines|regulations|compliance|institutional|animal\s+care|animal\s+welfare)\b", re.I),
            "institution_mention": re.compile(r"\b(university|institute|center|faculty|department|school|college|hospital|medical\s+center|research\s+center|laboratory|lab)\b", re.I),
        }
    
    def extract_sentences_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing specific keywords"""
        sentences = []
        sentence_endings = r'[.!?]+'
        text_sentences = re.split(sentence_endings, text)
        
        for sentence in text_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            # Check if any keyword is in the sentence
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', sentence, re.I):
                    sentences.append(sentence)
                    break
        
        return list(set(sentences))  # Remove duplicates
    
    def extract_institutions(self, text: str) -> List[str]:
        """Extract institution names from text"""
        institutions = []
        
        # Look for institution patterns
        institution_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|Institute|Center|Faculty|Department|School|College|Hospital|Laboratory|Lab))\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Medical\s+Center)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Research\s+Center)\b',
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, text)
            institutions.extend(matches)
        
        return list(set(institutions))  # Remove duplicates
    
    def find_matches_with_context(self, patterns: Dict[str, re.Pattern], text: str) -> Dict[str, List[str]]:
        """Find pattern matches and return matched strings"""
        hits = {}
        for label, pattern in patterns.items():
            found = pattern.findall(text)
            if found:
                # Normalize to strings
                found_str = []
                for f in found:
                    if isinstance(f, tuple):
                        found_str.append(" ".join([x for x in f if x]))
                    else:
                        found_str.append(f)
                hits[label] = list(set(found_str))  # Remove duplicates
        return hits
    
    def extract_ethics_sentences(self, text: str) -> List[str]:
        """Extract sentences containing ethics-related keywords"""
        sentences = []
        sentence_endings = r'[.!?]+'
        text_sentences = re.split(sentence_endings, text)
        
        for sentence in text_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            # Check if any ethics keyword is in the sentence
            for category, pattern in self.ethics_patterns.items():
                if pattern.search(sentence):
                    sentences.append(sentence)
                    break
        
        return list(set(sentences))  # Remove duplicates

    def analyze_ethics_evidence(self, text: str, sections: List[SectionText] = None) -> Dict[str, any]:
        """
        Ethics analysis with evidence extraction
        Returns sentences for better context
        """
        results = {
            "evidence_categories": {},
            "institutions_detected": [],
            "ethics_keywords": [],
            "evidence_sentences": [],
            "summary": ""
        }
        
        # Find ethics patterns
        ethics_hits = self.find_matches_with_context(self.ethics_patterns, text)
        
        evidence_categories = {}
        all_keywords = []
        all_sentences = []
        
        # Process each ethics category
        for category, keywords in ethics_hits.items():
            if keywords:
                # Extract sentences containing these keywords
                sentences = self.extract_sentences_with_keywords(text, keywords)
                
                evidence_categories[category] = {
                    "keywords_found": keywords,
                    "sentences": sentences,
                }
                
                all_keywords.extend(keywords)
                all_sentences.extend(sentences)
        
        # Extract institutions
        institutions = self.extract_institutions(text)
        
        # Extract ethics sentences
        ethics_sentences = self.extract_ethics_sentences(text)
        
        # Create summary
        summary_parts = []
        
        # Add institutions if found
        if institutions:
            summary_parts.append(f"Institutions: {', '.join(institutions[:3])}")  # Limit to first 3
        
        # Add key evidence sentences to summary
        if all_sentences:
            # Take first 3 sentences as summary
            key_sentences = all_sentences[:3]
            summary_parts.append(f"Evidence: {' | '.join(key_sentences)}")
        
        summary = "; ".join(summary_parts) if summary_parts else "No ethics evidence found"
        
        # Update results
        results.update({
            "evidence_categories": evidence_categories,
            "institutions_detected": institutions,
            "ethics_keywords": list(set(all_keywords)),
            "evidence_sentences": list(set(all_sentences)),
            "ethics_sentences": ethics_sentences,
            "summary": summary
        })
        
        return results
    
    def process_full_paper(self, doi: str) -> Dict[str, any]:
        """Process full paper text from PMC and return ethics analysis"""
        try:
            # Fetch full paper text from PMC
            paper_data = self.pmc_fetcher.fetch_full_paper_text(doi)
            
            if paper_data.success:
                # Analyze full paper text with section information
                return self.analyze_ethics_evidence(paper_data.full_text, paper_data.sections)
            else:
                # Fallback to empty result
                return {
                    "evidence_categories": {},
                    "institutions_detected": [],
                    "ethics_keywords": [],
                    "evidence_sentences": [],
                    "summary": f"No full paper available: {paper_data.error_message}"
                }
                
        except Exception as e:
            logging.warning(f"Full paper ethics analysis failed for {doi}: {e}")
            return {
                "evidence_categories": {},
                "institutions_detected": [],
                "ethics_keywords": [],
                "evidence_sentences": [],
                "summary": f"Error: {str(e)}"
            }
    
    def process_full_text(self, full_text: str) -> Dict[str, any]:
        """Process provided text and return ethics analysis (fallback method)"""
        if not full_text or full_text == "No text available":
            return {
                "evidence_categories": {},
                "institutions_detected": [],
                "ethics_keywords": [],
                "evidence_sentences": [],
                "summary": "No text available"
            }
        
        return self.analyze_ethics_evidence(full_text)
    
    def batch_process_texts(self, texts: List[str]) -> List[Dict[str, any]]:
        """Process multiple texts"""
        results = []
        for text in texts:
            results.append(self.process_full_text(text))
        return results
