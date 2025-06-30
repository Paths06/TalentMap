import streamlit as st
import pandas as pd
import re
import spacy
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STREAMLIT NEWSLETTER TALENT EXTRACTION DASHBOARD
# =============================================================================

st.set_page_config(
    page_title="Newsletter Talent Extraction",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load and cache models for better performance"""
    models = {}
    
    try:
        # Load spaCy model
        models['spacy'] = spacy.load('en_core_web_sm')
        st.success("✅ spaCy model loaded")
    except OSError:
        st.error("❌ spaCy model not found. Please install: python -m spacy download en_core_web_sm")
    
    try:
        # Load BERT NER model
        models['bert_ner'] = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("✅ BERT NER model loaded")
    except Exception as e:
        st.warning(f"⚠️ BERT NER failed to load: {e}")
    
    return models

class StreamlitTalentExtractor:
    def __init__(self, models):
        self.models = models
    
    def extract_with_spacy(self, text):
        """Extract entities using spaCy"""
        if 'spacy' not in self.models:
            return []
        
        doc = self.models['spacy'](text)
        extractions = []
        
        # Advanced regex patterns for financial newsletters
        patterns = [
            (r"(\w+\s+\w+)'s\s+([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners))\s+(?:will\s+)?(?:trade|launch|debut)", "launch"),
            (r"(\w+\s+\w+)\s+joins\s+([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners))", "hire"),
            (r"(\w+\s+\w+)\s+(?:picked|appointed|named|tapped)\s+(?:for|as|to).*?(?:at\s+)?([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners|Treasury|Bank))", "promotion"),
            (r"([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners))\s+(?:hires|appoints|taps|names)\s+(\w+\s+\w+)", "hire"),
            (r"(\w+\s+\w+)\s+(?:teams up|partners)\s+with.*?(?:on|forming|creating)\s+([A-Z][A-Za-z\s]*)", "partnership"),
            (r"(\w+\s+\w+)\s+(?:follows|following|replaces).*?(\w+\s+\w+)\s+departure", "hire"),
            (r"(\w+\s+\w+)\s+(?:eyes|preps|plots|readies)\s+([A-Z][A-Za-z\s]*(?:launch|debut))", "launch"),
            (r"([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners|Treasury))\s+promotes\s+(\w+\s+\w+)", "promotion"),
        ]
        
        for pattern, movement_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    name, company = self.classify_person_company(groups[0], groups[1])
                    
                    if name and self.is_valid_person_name(name):
                        extractions.append({
                            'Name': name,
                            'Company': company,
                            'Movement Type': movement_type,
                            'Confidence': 0.9,
                            'Method': 'spacy_regex',
                            'Raw Match': match.group(0)
                        })
        
        return extractions
    
    def extract_with_bert_ner(self, text):
        """Extract entities using BERT NER"""
        if 'bert_ner' not in self.models:
            return []
        
        try:
            entities = self.models['bert_ner'](text)
            extractions = []
            
            persons = [ent for ent in entities if ent['entity_group'] == 'PER']
            orgs = [ent for ent in entities if ent['entity_group'] == 'ORG']
            
            for person in persons:
                person_name = person['word'].strip()
                if self.is_valid_person_name(person_name):
                    person_start = person['start']
                    nearby_orgs = [
                        org for org in orgs 
                        if abs(org['start'] - person_start) < 200
                    ]
                    
                    if nearby_orgs:
                        best_org = min(nearby_orgs, key=lambda x: abs(x['start'] - person_start))
                        movement_type = self.determine_movement_type(
                            text[max(0, person_start-50):person_start+100]
                        )
                        
                        extractions.append({
                            'Name': person_name,
                            'Company': best_org['word'].strip(),
                            'Movement Type': movement_type,
                            'Confidence': round((person['score'] + best_org['score']) / 2, 2),
                            'Method': 'bert_ner',
                            'Raw Match': text[person_start:person_start+50]
                        })
            
            return extractions
        except Exception as e:
            st.error(f"BERT NER extraction failed: {e}")
            return []
    
    def classify_person_company(self, text1, text2):
        """Determine which text is a person name vs company name"""
        company_indicators = ['capital', 'management', 'fund', 'group', 'partners', 'treasury', 'bank']
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        text1_is_company = any(indicator in text1_lower for indicator in company_indicators)
        text2_is_company = any(indicator in text2_lower for indicator in company_indicators)
        
        if text1_is_company and not text2_is_company:
            return text2, text1
        elif text2_is_company and not text1_is_company:
            return text1, text2
        else:
            return text1, text2
    
    def is_valid_person_name(self, name):
        """Validate if text looks like a person name"""
        if not name or len(name.strip()) < 5:
            return False
        
        words = name.strip().split()
        if len(words) != 2:
            return False
        
        for word in words:
            if not word.isalpha() or not word[0].isupper():
                return False
        
        exclusions = [
            'capital', 'management', 'fund', 'group', 'treasury', 'bank',
            'former', 'senior', 'head', 'chief', 'director', 'manager'
        ]
        
        name_lower = name.lower()
        if any(exc in name_lower for exc in exclusions):
            return False
        
        return True
    
    def determine_movement_type(self, text):
        """Determine movement type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['joins', 'joined', 'joining', 'hires', 'hired']):
            return 'hire'
        elif any(word in text_lower for word in ['launches', 'launch', 'debut', 'preps', 'readies']):
            return 'launch'
        elif any(word in text_lower for word in ['promotes', 'promoted', 'appointed', 'named', 'picked']):
            return 'promotion'
        elif any(word in text_lower for word in ['forms', 'forming', 'partners', 'teams up']):
            return 'partnership'
        else:
            return 'movement'
    
    def process_newsletter(self, text):
        """Process newsletter with all methods"""
        all_extractions = []
        
        # Method 1: spaCy extraction
        spacy_results = self.extract_with_spacy(text)
        all_extractions.extend(spacy_results)
        
        # Method 2: BERT NER extraction
        bert_results = self.extract_with_bert_ner(text)
        all_extractions.extend(bert_results)
        
        # Deduplicate
        return self.deduplicate_extractions(all_extractions)
    
    def deduplicate_extractions(self, extractions):
        """Remove duplicates"""
        seen = set()
        unique_extractions = []
        
        for extraction in extractions:
            key = (extraction['Name'].lower(), extraction['Company'].lower())
            if key not in seen:
                seen.add(key)
                unique_extractions.append(extraction)
        
        return unique_extractions

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("🎯 Newsletter Talent Movement Extractor")
    st.markdown("### Extract talent movements from newsletters and export to CSV")
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload** your newsletter file (TXT format)
    2. **Or paste** newsletter text directly
    3. **Click** 'Extract Talent Movements'
    4. **Download** results as CSV
    
    **Supports:**
    - Hires and departures
    - Promotions and appointments
    - Fund launches
    - Company formations
    """)
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    extractor = StreamlitTalentExtractor(models)
    
    # Input methods
    st.header("📄 Input Newsletter")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Option 1: Upload File")
        uploaded_file = st.file_uploader(
            "Choose a TXT file",
            type=['txt'],
            help="Upload your newsletter in TXT format"
        )
    
    with col2:
        st.subheader("Option 2: Paste Text")
        pasted_text = st.text_area(
            "Paste newsletter content here",
            height=200,
            placeholder="Paste your newsletter text here..."
        )
    
    # Get text content
    newsletter_text = ""
    
    if uploaded_file is not None:
        newsletter_text = str(uploaded_file.read(), "utf-8")
        st.success(f"✅ File uploaded: {len(newsletter_text)} characters")
    elif pasted_text:
        newsletter_text = pasted_text
        st.info(f"📝 Text pasted: {len(newsletter_text)} characters")
    
    # Process button
    if st.button("🚀 Extract Talent Movements", type="primary"):
        if newsletter_text:
            with st.spinner("Extracting talent movements..."):
                try:
                    extractions = extractor.process_newsletter(newsletter_text)
                    
                    if extractions:
                        st.success(f"✅ Found {len(extractions)} talent movements!")
                        
                        # Create DataFrame
                        df = pd.DataFrame(extractions)
                        
                        # Display results
                        st.header("📊 Extracted Talent Movements")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"talent_movements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Statistics
                        st.header("📈 Extraction Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Movements", len(extractions))
                        
                        with col2:
                            hires = len([e for e in extractions if e['Movement Type'] == 'hire'])
                            st.metric("Hires", hires)
                        
                        with col3:
                            launches = len([e for e in extractions if e['Movement Type'] == 'launch'])
                            st.metric("Launches", launches)
                        
                        with col4:
                            promotions = len([e for e in extractions if e['Movement Type'] == 'promotion'])
                            st.metric("Promotions", promotions)
                        
                        # Movement type breakdown
                        movement_counts = df['Movement Type'].value_counts()
                        st.header("📋 Movement Type Breakdown")
                        st.bar_chart(movement_counts)
                        
                    else:
                        st.warning("⚠️ No talent movements found in the newsletter.")
                        st.info("💡 Try uploading a newsletter with more explicit talent movement information.")
                
                except Exception as e:
                    st.error(f"❌ Error processing newsletter: {str(e)}")
        else:
            st.error("❌ Please upload a file or paste newsletter text.")
    
    # Sample data section
    st.header("🧪 Test with Sample Data")
    if st.button("Try Sample Newsletter"):
        sample_text = """
        Harrison Balistreri's Inevitable Capital Management will trade l/s strat.
        Adnan Choudhury joins following Gregory Dunn departure.
        Daniel Crews picked for position at Tennessee Treasury.
        Sarah Gray joins Neil Chriss on forming Edge Peak.
        Robin Boldt to debut ROCK2 Capital in London.
        Centiva taps senior ExodusPoint PM for CRO.
        Dakota Wealth appoints FoHFs PM to CIO.
        """
        
        with st.spinner("Processing sample data..."):
            extractions = extractor.process_newsletter(sample_text)
            
            if extractions:
                st.success(f"✅ Sample processed: {len(extractions)} movements found!")
                df = pd.DataFrame(extractions)
                st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
