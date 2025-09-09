import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from text_fetcher import PaperFetcher, FullPaperText, SectionText

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
        self.paper_fetcher = PaperFetcher(tool_name=tool_name, email=email)
        
        # Ethics patterns - core concepts only
        self.ethics_patterns = {
            # Committees and boards
            "ethics_committee": re.compile(r"\b(ethics\s+committee|institutional\s+review\s+board|IRB|ethical\s+committee|research\s+ethics\s+committee|REC|animal\s+ethics\s+committee|AEC)\b", re.I),
            "animal_care": re.compile(r"\b(animal\s+care\s+and\s+use\s+committee|ACUC|IACUC|institutional\s+animal\s+care\s+and\s+use\s+committee|animal\s+welfare\s+committee|AWC)\b", re.I),
            "institutional_approval": re.compile(r"\b(institutional\s+approval|university\s+approval|faculty\s+approval|department\s+approval|institute\s+approval|center\s+approval)\b", re.I),
            
            # Guidelines and compliance
            "guidelines": re.compile(r"\b(NIH\s+Guide|ARRIVE|3R|replacement|animal\s+welfare)\b", re.I),
            "regulations": re.compile(r"\b(regulations|regulatory\s+compliance|ethical\s+protocol|animal\s+protocol)\b", re.I),
            
            # Approval and consent
            "approval": re.compile(r"\b(informed\s+consent|ethical\s+approval|ethical\s+permission)\b", re.I),
            
            # Declaration statements
            "declaration": re.compile(r"\b(conducted\s+in\s+accordance|in\s+accordance\s+with|following\s+guidelines)\b", re.I),
            "ethical_conduct": re.compile(r"\b(ethical\s+conduct)\b", re.I),
            
            # Animal welfare
            "animal_welfare": re.compile(r"\b(animal\s+welfare|humane\s+treatment|humane\s+care|animal\s+rights)\b", re.I),
            "minimize_suffering": re.compile(r"\b(minimize\s+suffering|reduce\s+pain|alleviate\s+distress|prevent\s+suffering|minimize\s+stress|reduce\s+discomfort|humane\s+endpoints)\b", re.I),
            
            # Institutions
            "institution_mention": re.compile(r"\b(university|institute|center|faculty|department|school|college|hospital|medical\s+center|research\s+center|laboratory|lab|animal\s+facility|central\s+animal\s+laboratory|CDL|vivarium|animal\s+research\s+facility)\b", re.I),
        }
        
        # Context patterns for sentence extraction
        self.context_patterns = {
            "ethics_section": re.compile(r"\b(ethics|ethical|welfare|approval|committee|guidelines|regulations|compliance|institutional|animal\s+care|animal\s+welfare)\b", re.I),
        }
    
    def extract_sentences_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing specific keywords"""
        sentences = []
        sentence_endings = r'[.!?]+'
        text_sentences = re.split(sentence_endings, text)
        
        for sentence in text_sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:  # Skip very short sentences
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
        
        # institution patterns for name extraction
        institution_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|Institute|Center|Faculty|Department|School|College|Hospital|Laboratory|Lab))\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Medical\s+Center)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Research\s+Center)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Animal\s+Facility)\b',
            r'\b(Central\s+Animal\s+Laboratory|CDL)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Vivarium)\b',
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, text)
            institutions.extend(matches)
        
        return list(set(institutions))  # Remove duplicates
    
    def extract_institution_sentences_with_sections(self, text: str, sections: List[SectionText] = None) -> List[str]:
        """Extract institution sentences with section information"""
        institution_sentences = []
        
        if sections:
            for section in sections:
                section_text = section.text
                section_name = section.section_name
                section_type = section.section_type
                
                # Only search in body and methods sections for institution mentions
                if section_type.lower() not in ['body', 'methods', 'methodology']:
                    continue
                
                sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
                sentences = re.split(sentence_endings, section_text)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 5:
                        continue
                    sentence = re.sub(r'\s+', ' ', sentence)
                    
                    # Check if any institution keyword is in the sentence
                    if self.ethics_patterns["institution_mention"].search(sentence):
                        formatted_entry = f"[{section_type}] {sentence}"
                        institution_sentences.append(formatted_entry)
        else:
            # Fallback to full text
            sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(sentence_endings, text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 5:
                    continue
                sentence = re.sub(r'\s+', ' ', sentence)
                
                # Check if any institution keyword is in the sentence
                if self.ethics_patterns["institution_mention"].search(sentence):
                    formatted_entry = f"[full_text] {sentence}"
                    institution_sentences.append(formatted_entry)
        
        # Remove duplicates
        seen_sentences = set()
        unique_sentences = []
        for item in institution_sentences:
            sentence_key = item.lower().strip()
            if sentence_key not in seen_sentences:
                seen_sentences.add(sentence_key)
                unique_sentences.append(item)
        
        return unique_sentences
    
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
    
    def extract_ethics_sentences_with_sections(self, text: str, sections: List[SectionText] = None) -> List[str]:
        """Extract ethics sentences with section information"""
        ethics_sentences = []
        
        if sections:
            for section in sections:
                section_text = section.text
                section_name = section.section_name
                section_type = section.section_type
                
                sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
                sentences = re.split(sentence_endings, section_text)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 5:
                        continue
                    sentence = re.sub(r'\s+', ' ', sentence)
                    
                    # Check if any ethics keyword is in the sentence (excluding institutions)
                    for category, pattern in self.ethics_patterns.items():
                        if category != "institution_mention" and pattern.search(sentence):
                            formatted_entry = f"[{section_type}] {sentence}"
                            ethics_sentences.append(formatted_entry)
                            break
        else:
            # Fallback to full text
            sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(sentence_endings, text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 5:
                    continue
                sentence = re.sub(r'\s+', ' ', sentence)
                
                # Check if any ethics keyword is in the sentence (excluding institutions)
                for category, pattern in self.ethics_patterns.items():
                    if category != "institution_mention" and pattern.search(sentence):
                        formatted_entry = f"[full_text] {sentence}"
                        ethics_sentences.append(formatted_entry)
                        break
        
        # Remove duplicates
        seen_sentences = set()
        unique_sentences = []
        for item in ethics_sentences:
            sentence_key = item.lower().strip()
            if sentence_key not in seen_sentences:
                seen_sentences.add(sentence_key)
                unique_sentences.append(item)
        
        return unique_sentences

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
        
        # Find ethics patterns (excluding institutions to avoid duplication)
        ethics_patterns_no_institutions = {k: v for k, v in self.ethics_patterns.items() if k != "institution_mention"}
        ethics_hits = self.find_matches_with_context(ethics_patterns_no_institutions, text)
        
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
        
        # Extract institution sentences with section information
        institution_sentences = self.extract_institution_sentences_with_sections(text, sections)
        
        # Extract ethics sentences with section information (excluding institutions)
        ethics_sentences_with_sections = self.extract_ethics_sentences_with_sections(text, sections)
        
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
            "institution_sentences": institution_sentences,
            "ethics_keywords": list(set(all_keywords)),
            "evidence_sentences": ethics_sentences_with_sections,
            "summary": summary
        })
        
        return results
    
    def process_full_paper(self, doi: str) -> Dict[str, any]:
        """Process full paper text from PMC and return ethics analysis"""
        try:
            # Fetch full paper text from PMC
            paper_data = self.paper_fetcher.fetch_full_paper_text(doi)
            
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
