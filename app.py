# =============================================================================
# 🚀 STREAMLIT APP USING YOUR TRAINED EXTRACTION SYSTEM
# =============================================================================
"""
Streamlit app that uses your trained_extraction_system.json
Deploy this on Streamlit Cloud for public use!

Files needed:
- streamlit_app.py (this file)
- trained_extraction_system.json (from Kaggle training)
- requirements.txt
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

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI-Trained Talent Tracker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
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
        st.info("Upload your trained system file from Kaggle training")
        return None

@st.cache_resource
def load_best_model(system):
    """Load the best performing model from training"""
    if not system or not system.get('best_combinations'):
        return None, None
    
    # Get best combination
    best_combo = system['best_combinations'][0]
    model_key = best_combo['model']
    prompt_key = best_combo['prompt']
    
    # Get Streamlit-compatible info
    streamlit_prompts = system.get('streamlit_prompts', {})
    prompt_config = streamlit_prompts.get(f"{model_key}_{prompt_key}")
    
    if not prompt_config:
        st.warning(f"⚠️ No configuration found for {model_key}_{prompt_key}")
        return None, None
    
    try:
        # Load model
        model_name = prompt_config['model_name']
        generation_config = prompt_config['generation_config']
        
        with st.spinner(f"Loading {model_name}..."):
            model = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1,
                **generation_config
            )
        
        return model, prompt_config
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None

# =============================================================================
# EXTRACTION ENGINE
# =============================================================================

class TrainedExtractor:
    def __init__(self, system, model, prompt_config):
        self.system = system
        self.model = model
        self.prompt_config = prompt_config
        self.fallback_patterns = system.get('fallback_patterns', [])
        self.validation_rules = system.get('validation_rules', {})
    
    def extract_movements(self, text):
        """Extract movements using trained system"""
        movements = []
        
        # Try LLM extraction first
        if self.model and self.prompt_config:
            llm_movements = self.extract_with_llm(text)
            movements.extend(llm_movements)
        
        # Try fallback patterns if LLM didn't find much
        if len(movements) < 3:
            pattern_movements = self.extract_with_patterns(text)
            movements.extend(pattern_movements)
        
        # Remove duplicates and validate
        unique_movements = self.deduplicate_and_validate(movements)
        
        return unique_movements
    
    def extract_with_llm(self, text):
        """Extract using the trained LLM"""
        movements = []
        
        # Split text into chunks for processing
        chunks = self.split_text_for_processing(text)
        
        for chunk in chunks[:5]:  # Process max 5 chunks
            try:
                # Use trained prompt template
                prompt = self.prompt_config['prompt_template'].format(text=chunk)
                
                # Generate response
                response = self.model(
                    prompt,
                    max_new_tokens=self.prompt_config['generation_config']['max_new_tokens'],
                    temperature=self.prompt_config['generation_config']['temperature'],
                    do_sample=self.prompt_config['generation_config']['do_sample'],
                    return_full_text=self.prompt_config['generation_config']['return_full_text'],
                    pad_token_id=50256
                )
                
                generated_text = response[0]['generated_text']
                
                # Parse response
                chunk_movements = self.parse_llm_response(generated_text)
                movements.extend(chunk_movements)
                
            except Exception as e:
                st.warning(f"⚠️ LLM extraction failed for chunk: {e}")
                continue
        
        return movements
    
    def extract_with_patterns(self, text):
        """Extract using fallback regex patterns"""
        movements = []
        
        for pattern, movement_type in self.fallback_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                person, company = self.extract_entities_from_groups(groups, movement_type, match.group(0))
                
                if person and company:
                    movements.append({
                        'name': person.strip(),
                        'company': company.strip(),
                        'movement_type': movement_type,
                        'confidence': 0.8,
                        'method': 'patterns',
                        'context': match.group(0)[:200]
                    })
        
        return movements
    
    def split_text_for_processing(self, text):
        """Split text into manageable chunks"""
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Split by paragraphs or sentences
        chunks = []
        paragraphs = cleaned_text.split('\n')
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 100 and self.has_talent_indicators(paragraph):
                chunks.append(paragraph.strip())
        
        return chunks
    
    def clean_text(self, text):
        """Clean newsletter text"""
        # Remove email headers, URLs, HTML
        text = re.sub(r'From:.*?\n|Sent:.*?\n|To:.*?\n|Subject:.*?\n', '', text)
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'urldefense\.proofpoint\.com[^\s]+', '', text)
        text = re.sub(r'Update email preferences.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Privacy Statement.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'________________________________+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def has_talent_indicators(self, text):
        """Check if text has talent movement indicators"""
        indicators = [
            'joins', 'hires', 'appoints', 'promotes', 'taps', 'named', 'picked',
            'launch', 'debut', 'founding', 'starting', 'preps', 'plots', 'eyes',
            'Capital', 'Management', 'Fund', 'CEO', 'CIO', 'COO', 'CFO'
        ]
        return any(indicator in text for indicator in indicators)
    
    def parse_llm_response(self, response):
        """Parse LLM response using trained patterns"""
        movements = []
        
        # Use multiple parsing strategies
        patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:at|joins?|started?)\s+([A-Z][A-Za-z\s]+?)\s*\(([^)]+)\)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+):\s*([A-Z][A-Za-z\s]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:joins|joined|starts|started|launches)\s+([A-Z][A-Za-z\s]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[-–—]\s*([A-Z][A-Za-z\s]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+).*?([A-Z][A-Za-z\s]*(?:Capital|Management|Fund|Group|Partners))',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2 and self.is_valid_name(groups[0]):
                    movements.append({
                        'name': groups[0].strip(),
                        'company': groups[1].strip(),
                        'movement_type': groups[2].strip().lower() if len(groups) > 2 else 'movement',
                        'confidence': 0.85,
                        'method': 'llm',
                        'context': match.group(0)
                    })
        
        return movements
    
    def extract_entities_from_groups(self, groups, movement_type, full_match):
        """Extract person and company from regex groups"""
        person, company = None, None
        
        if movement_type == "launch":
            if "'s" in full_match:
                person, company = groups[0], groups[1]
            else:
                person, company = groups[0], groups[1]
        elif movement_type in ["hire", "promotion"]:
            if "joins" in full_match:
                person, company = groups[0], groups[1]
            elif "hires" in full_match or "appoints" in full_match:
                company, person = groups[0], groups[1]
        
        return person, company
    
    def is_valid_name(self, name):
        """Validate person name using trained rules"""
        if not name or len(name) < self.validation_rules.get('min_name_length', 5):
            return False
        
        words = name.strip().split()
        if len(words) != 2:
            return False
        
        # Check alphabetic requirement
        if self.validation_rules.get('require_alphabetic', True):
            for word in words:
                if not word.isalpha() or not word[0].isupper():
                    return False
        
        # Check exclusion patterns
        exclude_patterns = self.validation_rules.get('exclude_patterns', [])
        name_lower = name.lower()
        for pattern in exclude_patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return False
        
        return True
    
    def deduplicate_and_validate(self, movements):
        """Remove duplicates and validate all movements"""
        seen = set()
        unique = []
        
        for movement in movements:
            # Validate first
            if not self.is_valid_name(movement.get('name', '')):
                continue
            
            # Check for duplicates
            key = (
                movement['name'].lower().strip(),
                movement['company'].lower().strip(),
                movement['movement_type']
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(movement)
        
        return unique

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("🎯 AI-Trained Talent Movement Tracker")
    st.markdown("**Powered by your custom-trained extraction system**")
    
    # Load trained system
    system = load_trained_system()
    if not system:
        st.stop()
    
    # Sidebar with training info
    st.sidebar.header("🧠 Training Information")
    
    training_stats = system.get('performance_metrics', {})
    st.sidebar.info(f"""
    **Training Platform:** {system.get('platform', 'Unknown')}  
    **Training Date:** {system.get('training_date', 'Unknown')[:10]}  
    **Models Trained:** {len(system.get('models_trained', []))}  
    **Best Model:** {system.get('performance_metrics', {}).get('recommended_model', 'Unknown')}  
    **Quality Score:** {training_stats.get('avg_quality_score', 0):.1%}
    """)
    
    if system.get('best_combinations'):
        st.sidebar.subheader("🏆 Best Combinations")
        for i, combo in enumerate(system['best_combinations'][:3], 1):
            st.sidebar.write(f"{i}. **{combo['model'].upper()}** + {combo['prompt']} ({combo['quality']:.1%})")
    
    # Load best model
    model, prompt_config = load_best_model(system)
    
    if not model:
        st.error("❌ Could not load trained model")
        st.info("💡 Using fallback patterns only")
    else:
        st.success(f"✅ Loaded trained model: **{prompt_config['model_name']}**")
    
    # File upload
    st.header("📁 Upload Newsletter Files")
    uploaded_files = st.file_uploader(
        "Upload newsletter .txt files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload financial newsletter files for talent movement extraction"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} files uploaded")
        
        # Extraction options
        col1, col2 = st.columns(2)
        with col1:
            use_llm = st.checkbox("🤖 Use trained LLM model", value=True, disabled=not model)
        with col2:
            use_patterns = st.checkbox("🔍 Use fallback patterns", value=True)
        
        if st.button("🚀 Extract Talent Movements", type="primary"):
            
            # Initialize extractor
            extractor = TrainedExtractor(system, model if use_llm else None, prompt_config if use_llm else None)
            
            all_movements = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                # Read content
                content = str(file.read(), "utf-8", errors="replace")
                
                # Extract movements
                movements = extractor.extract_movements(content)
                all_movements.extend(movements)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            
            # Store results
            st.session_state.movements = all_movements
            st.rerun()
    
    # Display results
    if 'movements' in st.session_state and st.session_state.movements:
        movements = st.session_state.movements
        
        st.header("📋 Extracted Movements")
        st.success(f"Found **{len(movements)}** talent movements!")
        
        # Create DataFrame
        df = pd.DataFrame(movements)
        
        # Add analysis columns
        df['asia_related'] = df['context'].str.contains(
            'asia|china|singapore|hong kong|japan|korea|india|apac', 
            case=False, na=False
        )
        df['senior_level'] = df['context'].str.contains(
            'ceo|cio|coo|cfo|chief|president|director|founder', 
            case=False, na=False
        )
        df['financial_firm'] = df['company'].str.contains(
            'capital|management|fund|group|partners|advisors', 
            case=False, na=False
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movements", len(df))
        with col2:
            st.metric("🌏 Asia-Related", len(df[df['asia_related']]))
        with col3:
            st.metric("👑 Senior-Level", len(df[df['senior_level']]))
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
        
        # Method breakdown
        st.subheader("🔧 Extraction Methods")
        method_counts = df['method'].value_counts()
        st.bar_chart(method_counts)
        
        # Movement type breakdown
        st.subheader("📈 Movement Types")
        type_counts = df['movement_type'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(type_counts)
        with col2:
            st.write("**Top Companies:**")
            company_counts = df['company'].value_counts().head(10)
            st.dataframe(company_counts)
        
        # Filters
        st.subheader("🔍 Filter Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            movement_filter = st.selectbox(
                "Movement Type",
                ['All'] + list(df['movement_type'].unique())
            )
        with col2:
            method_filter = st.selectbox(
                "Extraction Method", 
                ['All'] + list(df['method'].unique())
            )
        with col3:
            confidence_filter = st.slider(
                "Min Confidence",
                0.0, 1.0, 0.7, 0.1
            )
        
        # Apply filters
        filtered_df = df.copy()
        if movement_filter != 'All':
            filtered_df = filtered_df[filtered_df['movement_type'] == movement_filter]
        if method_filter != 'All':
            filtered_df = filtered_df[filtered_df['method'] == method_filter]
        filtered_df = filtered_df[filtered_df['confidence'] >= confidence_filter]
        
        # Show filtered results
        st.subheader(f"👥 Filtered Results ({len(filtered_df)} movements)")
        
        for i, row in filtered_df.iterrows():
            with st.expander(f"👤 {row['name']} → {row['company']} ({row['movement_type']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Name:**", row['name'])
                    st.write("**Company:**", row['company'])
                
                with col2:
                    st.write("**Movement:**", row['movement_type'].title())
                    st.write("**Method:**", row['method'])
                
                with col3:
                    st.write("**Confidence:**", f"{row['confidence']:.1%}")
                    
                    # Show flags
                    flags = []
                    if row['asia_related']:
                        flags.append("🌏 Asia")
                    if row['senior_level']:
                        flags.append("👑 Senior")
                    if row['financial_firm']:
                        flags.append("💰 Finance")
                    if row['confidence'] > 0.9:
                        flags.append("🎯 High Conf")
                    
                    if flags:
                        st.write("**Flags:**", " ".join(flags))
                
                if row.get('context'):
                    st.text_area("Context", row['context'][:300], height=60, disabled=True)
        
        # Download section
        st.header("💾 Download Results")
        
        # Prepare export
        export_df = prepare_export_data(filtered_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"talent_movements_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel with multiple sheets
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Movements', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total', 'Hires', 'Launches', 'Promotions', 'Asia-Related', 'Senior-Level'],
                    'Count': [
                        len(filtered_df),
                        len(filtered_df[filtered_df['movement_type'] == 'hire']),
                        len(filtered_df[filtered_df['movement_type'] == 'launch']),
                        len(filtered_df[filtered_df['movement_type'] == 'promotion']),
                        len(filtered_df[filtered_df['asia_related']]),
                        len(filtered_df[filtered_df['senior_level']])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"talent_movements_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def prepare_export_data(df):
    """Prepare data for export"""
    export_df = pd.DataFrame()
    
    export_df['Full Name'] = df['name']
    export_df['Company'] = df['company']
    export_df['Movement Type'] = df['movement_type'].str.title()
    export_df['Confidence Score'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    export_df['Extraction Method'] = df['method'].str.title()
    export_df['Asia Related'] = df['asia_related'].map({True: 'Yes', False: 'No'})
    export_df['Senior Level'] = df['senior_level'].map({True: 'Yes', False: 'No'})
    export_df['Financial Firm'] = df['financial_firm'].map({True: 'Yes', False: 'No'})
    export_df['Date Extracted'] = datetime.now().strftime('%Y-%m-%d')
    export_df['Source'] = 'AI-Trained System'
    
    return export_df

if __name__ == "__main__":
    main()
