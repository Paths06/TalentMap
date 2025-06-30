# =============================================================================
# 🚀 IMPROVED LLM STREAMLIT APP - BETTER PROMPTS & PARSING
# =============================================================================
"""
Fixed version with improved prompting and parsing for better results
"""

import streamlit as st
import pandas as pd
import json
import re
import torch
from datetime import datetime
from transformers import pipeline
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI-Trained Talent Tracker",
    page_icon="🎯",
    layout="wide"
)

# =============================================================================
# LOAD TRAINED SYSTEM
# =============================================================================

@st.cache_data
def load_trained_system():
    """Load the trained extraction system"""
    try:
        with open('trained_extraction_system.json', 'r') as f:
            system = json.load(f)
        return system
    except FileNotFoundError:
        st.error("❌ trained_extraction_system.json not found!")
        return None

@st.cache_resource
def load_best_model(system):
    """Load the best performing model with improved settings"""
    if not system or not system.get('best_combinations'):
        return None, None
    
    # Get best combination
    best_combo = system['best_combinations'][0]
    model_name = "distilgpt2"  # Use the best performing model
    
    try:
        with st.spinner(f"🤖 Loading {model_name}... (30-60 seconds)"):
            model = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=True,
                return_full_text=False,
                pad_token_id=50256
            )
        
        st.success(f"✅ Loaded {model_name} successfully!")
        return model, best_combo
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None, None

# =============================================================================
# IMPROVED EXTRACTION ENGINE
# =============================================================================

class ImprovedExtractor:
    def __init__(self, system, model=None, model_config=None):
        self.system = system
        self.model = model
        self.model_config = model_config
        self.fallback_patterns = system.get('fallback_patterns', [])
        self.validation_rules = system.get('validation_rules', {})
    
    def extract_movements(self, text):
        """Extract movements using improved approach"""
        all_movements = []
        
        # Always run patterns first (reliable baseline)
        pattern_movements = self.extract_with_patterns(text)
        all_movements.extend(pattern_movements)
        st.info(f"🔍 Pattern extraction: {len(pattern_movements)} movements")
        
        # Try LLM with improved prompts
        if self.model:
            llm_movements = self.extract_with_improved_llm(text)
            all_movements.extend(llm_movements)
            st.info(f"🤖 LLM extraction: {len(llm_movements)} movements")
        
        # Remove duplicates and validate
        unique_movements = self.deduplicate_and_validate(all_movements)
        st.success(f"✅ Total unique movements: {len(unique_movements)}")
        
        return unique_movements
    
    def extract_with_improved_llm(self, text):
        """Extract using improved LLM prompts"""
        movements = []
        
        # Split text into better chunks
        chunks = self.create_smart_chunks(text)
        
        for i, chunk in enumerate(chunks[:4]):  # Process up to 4 chunks
            try:
                # Use multiple improved prompts
                for prompt_type in ['direct', 'structured', 'examples']:
                    chunk_movements = self.extract_with_prompt_type(chunk, prompt_type)
                    movements.extend(chunk_movements)
                
            except Exception as e:
                st.warning(f"⚠️ LLM chunk {i+1} failed: {str(e)}")
                continue
        
        return movements
    
    def extract_with_prompt_type(self, text, prompt_type):
        """Extract using specific prompt type"""
        # Improved prompts based on what works
        prompts = {
            'direct': f"""Find people and companies in this text:

{text}

List each person like this:
John Smith - ABC Capital
Jane Doe - XYZ Fund

People:
""",
            
            'structured': f"""Extract talent movements from this newsletter:

{text}

Format each as:
Name: [First Last]
Company: [Company Name]

Results:
""",
            
            'examples': f"""Extract people like these examples:
"Harrison Balistreri's Inevitable Capital" = Harrison Balistreri - Inevitable Capital
"Daniel Crews picked for position" = Daniel Crews - Company
"Sarah Gray joins Neil Chriss" = Sarah Gray - Neil Chriss

Text: {text}

Extract:
"""
        }
        
        prompt = prompts.get(prompt_type, prompts['direct'])
        
        try:
            response = self.model(
                prompt,
                max_new_tokens=120,
                temperature=0.2,
                do_sample=True,
                return_full_text=False,
                pad_token_id=50256
            )
            
            generated_text = response[0]['generated_text']
            return self.parse_llm_response_improved(generated_text)
            
        except Exception as e:
            st.warning(f"⚠️ Prompt type {prompt_type} failed: {str(e)}")
            return []
    
    def create_smart_chunks(self, text):
        """Create better text chunks for processing"""
        # Clean text first
        cleaned = self.clean_text(text)
        
        # Split by common newsletter delimiters
        sections = re.split(r'\n\s*\n|\n\d{1,2}\s+(?:Jun|June|Jul|July|Aug|Sept|Sep)\s+\d{4}|_{10,}', cleaned)
        
        chunks = []
        for section in sections:
            section = section.strip()
            
            # Only keep sections with talent indicators and reasonable length
            if (len(section) > 100 and 
                len(section) < 1000 and 
                self.has_talent_indicators(section)):
                chunks.append(section)
        
        return chunks[:6]  # Max 6 chunks
    
    def parse_llm_response_improved(self, response):
        """Improved parsing of LLM responses"""
        movements = []
        
        # Multiple parsing strategies
        parsing_patterns = [
            # Name - Company format
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[-–—]\s*([A-Z][A-Za-z\s&]+)',
            
            # Name: Company format
            r'Name:\s*([A-Z][a-z]+\s+[A-Z][a-z]+).*?Company:\s*([A-Z][A-Za-z\s&]+)',
            
            # Direct mention format
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:at|joins?|from)\s+([A-Z][A-Za-z\s&]+)',
            
            # Capital/Fund pattern
            r'([A-Z][a-z]+\s+[A-Z][a-z]+).*?([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners))',
            
            # List format
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[:\-]\s*([A-Z][A-Za-z\s&]+)',
        ]
        
        for pattern in parsing_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    name = groups[0].strip()
                    company = groups[1].strip()
                    
                    # Clean company name
                    company = re.sub(r'\s*\(.*?\)\s*', '', company)  # Remove parentheses
                    company = company.split('\n')[0].strip()  # Take first line only
                    
                    if self.is_valid_extraction(name, company):
                        movements.append({
                            'name': name,
                            'company': company,
                            'movement_type': 'movement',
                            'confidence': 0.85,
                            'method': 'llm_improved',
                            'context': match.group(0)
                        })
        
        return movements
    
    def extract_with_patterns(self, text):
        """Extract using enhanced pattern matching"""
        movements = []
        cleaned_text = self.clean_text(text)
        
        # Enhanced patterns based on With Intelligence format
        enhanced_patterns = [
            # Possessive launches
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)'s\s+([A-Z][A-Za-z\s]+(?:Capital|Management|Fund|Group|Partners))", "launch"),
            
            # Joins patterns
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+joins\s+([A-Z][A-Za-z\s&]+?)(?:\s+as|\s+to|\s+following|\s*,|\s*$)", "hire"),
            
            # Hires/appoints patterns
            (r"([A-Z][A-Za-z\s&]+?)\s+(?:hires|appoints|taps)\s+(?:senior\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)", "hire"),
            
            # Picked for position
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+picked\s+for\s+position", "promotion"),
            
            # Promotes patterns
            (r"([A-Z][A-Za-z\s&]+?)\s+promotes\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", "promotion"),
            
            # Set to debut / launch patterns
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:set\s+to\s+debut|to\s+debut|plots|preps|eyes)\s+([A-Z][A-Za-z\s&]+)", "launch"),
            
            # Teams up / forming patterns
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:teams\s+up\s+with.*?on\s+forming|joins.*?on\s+forming)\s+([A-Z][A-Za-z\s&]+)", "hire"),
            
            # Professional title patterns
            (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+to\s+run\s+new\s+([A-Z][A-Za-z\s,&]+?)\s+book", "hire"),
        ]
        
        for pattern, movement_type in enhanced_patterns:
            try:
                matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    person, company = self.extract_entities_from_groups(groups, movement_type, match.group(0))
                    
                    if person and company and self.is_valid_extraction(person, company):
                        movements.append({
                            'name': person.strip(),
                            'company': company.strip(),
                            'movement_type': movement_type,
                            'confidence': 0.9,
                            'method': 'patterns_enhanced',
                            'context': match.group(0)[:300]
                        })
                        
            except Exception as e:
                continue
        
        return movements
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        # Remove email headers and footers
        text = re.sub(r'From:.*?\n|Sent:.*?\n|To:.*?\n|Subject:.*?\n', '', text)
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'urldefense\.proofpoint\.com[^\s]+', '', text)
        text = re.sub(r'Update email preferences.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Privacy Statement.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'________________________________+', '', text)
        text = re.sub(r'View article.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'View manager.*$', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def has_talent_indicators(self, text):
        """Check for talent movement indicators"""
        indicators = [
            'joins', 'hires', 'appoints', 'promotes', 'taps', 'named', 'picked',
            'launch', 'debut', 'founding', 'starting', 'preps', 'plots', 'eyes',
            'Capital', 'Management', 'Fund', 'CEO', 'CIO', 'COO', 'CFO', 'director',
            'teams up', 'forming', 'run new', 'set to debut'
        ]
        return any(indicator in text for indicator in indicators)
    
    def extract_entities_from_groups(self, groups, movement_type, full_match):
        """Extract person and company from regex groups"""
        person, company = None, None
        
        try:
            if movement_type == "launch":
                if "'s" in full_match:
                    person, company = groups[0], groups[1]
                elif "to debut" in full_match or "plots" in full_match:
                    person, company = groups[0], groups[1]
                    
            elif movement_type == "hire":
                if "joins" in full_match:
                    person, company = groups[0], groups[1]
                elif "hires" in full_match or "appoints" in full_match or "taps" in full_match:
                    company, person = groups[0], groups[1]
                elif "teams up" in full_match or "forming" in full_match:
                    person, company = groups[0], groups[1]
                elif "to run" in full_match:
                    person, company = groups[0], "Trading Desk"
                    
            elif movement_type == "promotion":
                if "promotes" in full_match:
                    company, person = groups[0], groups[1]
                elif "picked" in full_match:
                    person = groups[0]
                    company = self.find_company_in_context(full_match)
                    
        except Exception as e:
            pass
        
        return person, company
    
    def find_company_in_context(self, match_text):
        """Find company in context"""
        company_patterns = [
            r'([A-Z][A-Za-z\s]{3,30}(?:Capital|Management|Fund|Group|Partners|Treasury|Bank))',
            r'([A-Z][A-Za-z\s]{3,20}(?:LLC|Inc|Corp))'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, match_text)
            if matches:
                return matches[0]
        
        return "Company"
    
    def is_valid_extraction(self, name, company):
        """Enhanced validation"""
        # Validate name
        if not self.is_valid_name(name):
            return False
        
        # Validate company
        if not company or len(company.strip()) < 3:
            return False
        
        # Check if company looks legitimate
        company_lower = company.lower()
        invalid_companies = ['unknown', 'company', 'firm', 'group']
        if any(invalid in company_lower for invalid in invalid_companies):
            return False
        
        return True
    
    def is_valid_name(self, name):
        """Enhanced name validation"""
        if not name or len(name) < 5:
            return False
        
        words = name.strip().split()
        if len(words) != 2:
            return False
        
        # Check each word is alphabetic and capitalized
        for word in words:
            if not word.isalpha() or not word[0].isupper():
                return False
        
        # Enhanced exclusions
        exclusions = [
            'pm', 'ceo', 'cio', 'cfo', 'coo', 'cto', 'cmo', 'director', 'manager', 
            'head', 'chief', 'president', 'partner', 'analyst', 'associate',
            'capital', 'management', 'fund', 'group', 'partners', 'advisors', 
            'treasury', 'wealth', 'bank', 'securities', 'trading', 'investments',
            'pro', 'vet', 'alum', 'exec', 'senior', 'former', 'ex', 'staff',
            'team', 'board', 'committee', 'portfolio', 'hedge'
        ]
        
        name_lower = name.lower()
        if any(exc in name_lower for exc in exclusions):
            return False
        
        return True
    
    def deduplicate_and_validate(self, movements):
        """Remove duplicates and validate"""
        seen = set()
        unique = []
        
        for movement in movements:
            if not movement.get('name') or not movement.get('company'):
                continue
                
            # Create key for deduplication
            key = (
                movement['name'].lower().strip(),
                movement['company'].lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(movement)
        
        return unique

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("🎯 Improved AI Talent Tracker")
    st.markdown("**Enhanced extraction with better prompts and parsing**")
    
    # Load system
    system = load_trained_system()
    if not system:
        st.stop()
    
    # Sidebar info
    st.sidebar.header("🧠 System Information")
    st.sidebar.info(f"""
    **Platform:** {system.get('platform', 'Kaggle')}  
    **Models:** {len(system.get('models_trained', []))}  
    **Best Model:** DistilGPT2  
    **Enhancement:** Improved prompts & parsing
    """)
    
    # Load model
    model, model_config = load_best_model(system)
    
    if model:
        st.success(f"✅ Model loaded with enhanced prompting")
    else:
        st.warning("⚠️ Using pattern-only extraction")
    
    # File upload
    st.header("📁 Upload Newsletter Files")
    uploaded_files = st.file_uploader(
        "Upload .txt files",
        type=['txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} files uploaded")
        
        if st.button("🚀 Extract Movements", type="primary"):
            extractor = ImprovedExtractor(system, model, model_config)
            
            all_movements = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    content = str(file.read(), "utf-8", errors="replace")
                    movements = extractor.extract_movements(content)
                    all_movements.extend(movements)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.movements = all_movements
            st.rerun()
    
    # Display results
    if 'movements' in st.session_state and st.session_state.movements:
        movements = st.session_state.movements
        
        st.header(f"📋 Extracted {len(movements)} Movements")
        
        if movements:
            df = pd.DataFrame(movements)
            
            # Method breakdown
            if 'method' in df.columns:
                method_counts = df['method'].value_counts()
                st.subheader("🔧 Extraction Methods")
                st.bar_chart(method_counts)
            
            # Show movements
            st.subheader("👥 Individual Movements")
            for i, movement in enumerate(movements):
                with st.expander(f"👤 {movement.get('name', 'Unknown')} → {movement.get('company', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {movement.get('name', 'Unknown')}")
                        st.write(f"**Company:** {movement.get('company', 'Unknown')}")
                    
                    with col2:
                        st.write(f"**Type:** {movement.get('movement_type', 'unknown').title()}")
                        st.write(f"**Method:** {movement.get('method', 'unknown')}")
                        st.write(f"**Confidence:** {movement.get('confidence', 0):.1%}")
                    
                    if movement.get('context'):
                        st.text_area("Context", movement['context'][:300], height=70, disabled=True, key=f"ctx_{i}")
            
            # Download
            st.header("💾 Download Results")
            export_df = pd.DataFrame(movements)
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv,
                f"improved_extractions_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.warning("No movements found. Try uploading different newsletter files.")

if __name__ == "__main__":
    main()
