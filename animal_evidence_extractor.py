import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from pmc_text_fetcher import PMCTextFetcher, FullPaperText, SectionText

@dataclass
class InVivoEvidence:
    """Structured evidence for in vivo animal studies"""
    category: str
    keywords_found: List[str]
    sentences: List[str]
    section_sources: List[str]  # Which sections the evidence came from

class InVivoDetector:
    """
    In vivo analysis with evidence extraction
    Uses full paper text from PMC for better analysis
    """
    
    def __init__(self, tool_name: str = "animal_evidence_extractor", email: str = "hi.hoi@mail.nl"):
        # Initialize PMC text fetcher
        self.pmc_fetcher = PMCTextFetcher(tool_name=tool_name, email=email)
        
        # Species patterns
        self.species_patterns = {
            "rat": re.compile(r"\brat(s)?\b", re.I),
            "mouse": re.compile(r"\b(mouse|mice)\b", re.I),
            "zebrafish": re.compile(r"\b(zebrafish|Danio\s+rerio)\b", re.I),
            "rabbit": re.compile(r"\brabbit(s)?\b", re.I),
            "pig": re.compile(r"\b(pig(s)?|porcine|swine)\b", re.I),
            "chicken": re.compile(r"\b(chicken(s)?|Gallus\s+gallus|avian|poultry)\b", re.I),
            "nonhuman_primate": re.compile(r"\b(macaque|marmoset|monkey|non[-\s]?human\s+primate|rhesus|cynomolgus)\b", re.I),
            "dog": re.compile(r"\b(dog(s)?|canine|beagle)\b", re.I),
            "cat": re.compile(r"\b(cat(s)?|feline)\b", re.I),
            "sheep": re.compile(r"\b(sheep|ovine)\b", re.I),
            "cow": re.compile(r"\b(cow(s)?|bovine|cattle)\b", re.I),
            "horse": re.compile(r"\b(horse(s)?|equine)\b", re.I),
            "fish": re.compile(r"\b(fish|teleost|salmon|trout)\b", re.I),
            "frog": re.compile(r"\b(frog(s)?|Xenopus|amphibian)\b", re.I),
            "bird": re.compile(r"\b(bird(s)?|avian|pigeon|sparrow)\b", re.I),
            "reptile": re.compile(r"\b(reptile(s)?|lizard|snake|turtle)\b", re.I),
        }
        
        # Strain patterns
        self.strain_patterns = {
            "Sprague-Dawley": re.compile(r"\bSprague[-\s]?Dawley\b", re.I),
            "Wistar": re.compile(r"\bWistar\b", re.I),
            "Long-Evans": re.compile(r"\bLong[-\s]?Evans\b", re.I),
            "C57BL/6": re.compile(r"\bC57BL/?6\b", re.I),
            "BALB/c": re.compile(r"\bBALB/?c\b", re.I),
            "rTg-DI": re.compile(r"\brTg[-\s]?DI\b", re.I),
            "CD1": re.compile(r"\bCD1\b", re.I),
            "ICR": re.compile(r"\bICR\b", re.I),
        }
        
        # In vivo patterns - strict context
        self.in_vivo_patterns = {
            # Explicit in vivo mentions
            "explicit_in_vivo": re.compile(r"\bin\s?vivo\b", re.I),
            "in_vivo_study": re.compile(r"\bin\s?vivo\s+stud(?:y|ies)\b", re.I),
            "in_vivo_experiment": re.compile(r"\bin\s?vivo\s+experiment(s)?\b", re.I),
            "in_vivo_model": re.compile(r"\bin\s?vivo\s+model(s)?\b", re.I),
            
            # Animal procedures
            "anesthesia": re.compile(r"\banesthet(i[sz]ed|ia|ic)\b", re.I),
            "euthanasia": re.compile(r"\b(euthanis|euthaniz|sacrific|killed)\w*\b", re.I),
            "injections": re.compile(r"\b(intraperitoneal|i\.p\.|intravenous|i\.v\.|subcutaneous|s\.c\.|intramuscular|i\.m\.|intracerebral|i\.c\.|intracranial|intrahippocampal|intrastriatal|intraventricular)\s+injection\b", re.I),
            "surgery": re.compile(r"\b(surgery|surgical|stereotaxic|stereotactic|craniotomy|perfusion|transplantation|implantation|catheterization|cannulation|tracheotomy|laparotomy|thoracotomy)\b", re.I),
            
            # Behavioral tests
            "behavioral_tests": re.compile(r"\b(Morris\s+water\s+maze|open\s+field|elevated\s+plus\s+maze|rotarod|T\s?maze|radial\s+arm\s+maze|novel\s+object\s+recognition|fear\s+conditioning|passive\s+avoidance|active\s+avoidance|social\s+interaction|prepulse\s+inhibition|startle\s+response|grip\s+strength|balance\s+beam|wire\s+hanging)\b", re.I),
            
            # Physiological measurements
            "physiological": re.compile(r"\b(blood\s+pressure|heart\s+rate|respiratory\s+rate|body\s+temperature|weight\s+gain|weight\s+loss|food\s+intake|water\s+intake|urine\s+output|fecal\s+output|glucose\s+level|insulin\s+level|cholesterol|triglycerides|cytokines|inflammatory\s+markers)\s+(measurement|monitoring|assessment|analysis)\b", re.I),
            
            # Sampling procedures
            "sampling": re.compile(r"\b(cerebrospinal\s+fluid|CSF|blood\s+collection|tail\s+vein|cardiac\s+puncture|orbital\s+bleeding|jugular\s+vein|carotid\s+artery|femoral\s+vein|saphenous\s+vein|urine\s+collection|fecal\s+collection|tissue\s+biopsy|organ\s+removal|brain\s+extraction|liver\s+extraction|kidney\s+extraction|spleen\s+extraction|lung\s+extraction|heart\s+extraction)\b", re.I),
            
            # Animal husbandry
            "husbandry": re.compile(r"\b(housed|housing|cage|caging|temperature[-\s]controlled|humidity[-\s]controlled|12[-\s]?h\s+light|12[-\s]?h\s+dark|light[-\s]cycle|dark[-\s]cycle|diet|feeding|water|bedding|enrichment|social\s+isolation|group\s+housed|single\s+housed|pair\s+housed)\b", re.I),
            
            # Ethics and regulations
            "ethics": re.compile(r"\b(IACUC|ARRIVE|Institutional\s+Animal\s+Care|ethics\s+committee|animal\s+care\s+and\s+use|animal\s+welfare|3R|replacement|reduction|refinement|animal\s+protocol|animal\s+approval|ethical\s+approval|regulatory\s+compliance)\b", re.I),
            
            # Experimental design
            "experimental_design": re.compile(r"\b(randomized|randomization|blinded|blind|double[-\s]blind|single[-\s]blind|control\s+group|treatment\s+group|sham\s+control|vehicle\s+control|baseline|post[-\s]treatment|pre[-\s]treatment|follow[-\s]up|longitudinal|cross[-\s]sectional|cohort|intervention|manipulation)\b", re.I),
            
            # Disease models
            "disease_models": re.compile(r"\b(disease\s+model|animal\s+model|transgenic|knockout|knock[-\s]in|overexpression|mutant|mutagenesis|carcinogen|tumor\s+induction|infection\s+model|injury\s+model|trauma\s+model|stroke\s+model|diabetes\s+model|obesity\s+model|hypertension\s+model|asthma\s+model|arthritis\s+model|depression\s+model|anxiety\s+model|Alzheimer\s+model|Parkinson\s+model|Huntington\s+model|ALS\s+model)\b", re.I),
            
            # Drug administration
            "drug_administration": re.compile(r"\b(drug\s+administration|treatment|dosing|dose|concentration|mg/kg|μg/kg|ng/kg|μmol/kg|nmol/kg|oral\s+administration|gavage|intraperitoneal\s+injection|intravenous\s+injection|subcutaneous\s+injection|intramuscular\s+injection|topical\s+application|inhalation|intranasal|intracerebral|intraventricular|chronic\s+treatment|acute\s+treatment|repeated\s+dosing|single\s+dose|multiple\s+doses)\b", re.I),
            
            # Imaging and monitoring
            "imaging": re.compile(r"\b(MRI|fMRI|PET|CT|ultrasound|X[-\s]ray|radiography|microscopy|confocal|fluorescence|bioluminescence|fluorescence\s+imaging|live\s+imaging|real[-\s]time\s+imaging|monitoring|telemetry|implanted\s+sensor|wireless\s+monitoring|video\s+tracking|movement\s+tracking|activity\s+monitoring)\b", re.I),
            
            # Tissue analysis
            "tissue_analysis": re.compile(r"\b(histology|histological|histopathology|immunohistochemistry|IHC|immunofluorescence|IF|Western\s+blot|PCR|qPCR|RT[-\s]PCR|RNA\s+sequencing|DNA\s+sequencing|microarray|proteomics|metabolomics|transcriptomics|genomics|ELISA|flow\s+cytometry|FACS|microscopy|electron\s+microscopy|EM|TEM|SEM)\b", re.I),
        }
        
        # Context patterns for sentence extraction
        self.context_patterns = {
            "animal_mention": re.compile(r"\b(animal|animals|rodent|rodents|mammal|mammals)\b", re.I),
            "experiment_mention": re.compile(r"\b(experiment|experimental|study|studies|investigation|analysis)\b", re.I),
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
    
    def extract_species_sentences(self, text: str) -> List[str]:
        """Extract sentences containing species mentions"""
        sentences = []
        sentence_endings = r'[.!?]+'
        text_sentences = re.split(sentence_endings, text)
        
        for sentence in text_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            # Check if any species is in the sentence
            for species_name in self.species_patterns.keys():
                if re.search(rf'\b{re.escape(species_name)}\b', sentence, re.I):
                    sentences.append(sentence)
                    break
        
        return list(set(sentences))  # Remove duplicates

    def analyze_in_vivo_evidence(self, text: str, sections: List[SectionText] = None) -> Dict[str, any]:
        """
        In vivo analysis with evidence extraction
        Returns sentences for better context
        """
        results = {
            "evidence_categories": {},
            "species_detected": [],
            "strains_detected": [],
            "in_vivo_keywords": [],
            "evidence_sentences": [],
            "species_sentences": [],
            "summary": ""
        }
        
        # Find species patterns and extract sentences
        species_hits = self.find_matches_with_context(self.species_patterns, text)
        species_detected = list(species_hits.keys())
        species_sentences = self.extract_species_sentences(text)
        
        # Find strain patterns
        strain_hits = self.find_matches_with_context(self.strain_patterns, text)
        strains_detected = list(strain_hits.keys())
        
        # Find in vivo patterns
        in_vivo_hits = self.find_matches_with_context(self.in_vivo_patterns, text)
        
        evidence_categories = {}
        all_keywords = []
        all_sentences = []
        
        # Process each in vivo category
        for category, keywords in in_vivo_hits.items():
            if keywords:
                sentences = self.extract_sentences_with_keywords(text, keywords)
                evidence_categories[category] = {
                    "keywords_found": keywords,
                    "sentences": sentences,
                }
                all_keywords.extend(keywords)
                all_sentences.extend(sentences)
        
        # Create summary
        summary_parts = []
        if species_detected:
            summary_parts.append(f"Species: {', '.join(species_detected)}")
        if strains_detected:
            summary_parts.append(f"Strains: {', '.join(strains_detected)}")
        
        if all_sentences:
            key_sentences = all_sentences[:3]
            summary_parts.append(f"Evidence: {' | '.join(key_sentences)}")
        
        summary = "; ".join(summary_parts) if summary_parts else "No in vivo evidence found"
        
        results.update({
            "evidence_categories": evidence_categories,
            "species_detected": species_detected,
            "strains_detected": strains_detected,
            "in_vivo_keywords": list(set(all_keywords)),
            "evidence_sentences": list(set(all_sentences)),
            "species_sentences": species_sentences,
            "summary": summary
        })
        
        return results
    
    def process_abstract(self, abstract: str) -> Dict[str, any]:
        """Process abstract text (fallback method)"""
        if not abstract or abstract == "No abstract available":
            return {
                "evidence_categories": {},
                "species_detected": [],
                "strains_detected": [],
                "in_vivo_keywords": [],
                "evidence_sentences": [],
                "species_sentences": [],
                "summary": "No abstract available"
            }
        
        return self.analyze_in_vivo_evidence(abstract)
    
    def process_full_paper(self, doi: str) -> Dict[str, any]:
        """Process full paper text from PMC and return in vivo analysis"""
        try:
            # Fetch full paper text from PMC
            paper_data = self.pmc_fetcher.fetch_full_paper_text(doi)
            
            if paper_data.success:
                # Analyze full paper text with section information
                return self.analyze_in_vivo_evidence(paper_data.full_text, paper_data.sections)
            else:
                # Fallback to empty result
                return {
                    "evidence_categories": {},
                    "species_detected": [],
                    "strains_detected": [],
                    "in_vivo_keywords": [],
                    "evidence_sentences": [],
                    "species_sentences": [],
                    "summary": f"No full paper available: {paper_data.error_message}"
                }
                
        except Exception as e:
            logging.warning(f"Full paper in vivo analysis failed for {doi}: {e}")
            return {
                "evidence_categories": {},
                "species_detected": [],
                "strains_detected": [],
                "in_vivo_keywords": [],
                "evidence_sentences": [],
                "species_sentences": [],
                "summary": f"Error: {str(e)}"
            }
    
    def batch_process_abstracts(self, abstracts: List[str]) -> List[Dict[str, any]]:
        """Process multiple abstracts"""
        results = []
        for abstract in abstracts:
            results.append(self.process_abstract(abstract))
        return results
