# =============================================================================
# 🤖 LLM-POWERED STREAMLIT APP - PRODUCTION READY
# =============================================================================
"""
Streamlit app using your trained LLM models from Kaggle
- DistilGPT2 (best model from your training)
- GPT-Neo 1.3B (fallback)
- Trained patterns (final fallback)
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
    page_title="🤖 AI-Trained Talent Tracker",
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
        st.info("Make sure your trained system file is in the app directory")
        return None

@st.cache_resource
def load_llm_models(system):
    """Load LLM models with fallback strategy"""
    if not system or not system.get('streamlit_prompts'):
        return {}
    
    models = {}
    model_configs = system['streamlit_prompts']
    
    # Try to load models in order of performance
    model_priority = ['distilgpt2_examples', 'gpt_neo_examples', 'gpt2_medium_examples']
    
    for model_key in model_priority:
        if model_key in model_configs:
            config = model_configs[model_key]
            model_name = config['model_name']
            
            try:
                with st.spinner(f"Loading {model_name}..."):
                    # Load with optimized settings for Streamlit Cloud
                    model = pipeline(
                        "text-generation",
                        model=model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device=0 if torch.cuda.is_available() else -1,
                        trust_remote_code=True,
                        **config['generation_config']
                    )
                    
                    models[model_key] = {
                        'pipeline': model,
                        'config': config,
                        'model_name': model_name
                    }
                    
                    st.success(f"✅ Loaded {model_name}")
                    break  # Use first successfully loaded model
                    
            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_name}: {str(e)[:100]}...")
                continue
    
    if not models:
        st.error("❌ Could not load any LLM models")
        st.info("💡 App will use trained patterns as fallback")
    
    return models

# =============================================================================
# ENHANCED EXTRACTOR WITH LLM + PATTERNS
# =============================================================================

class EnhancedExtractor:
    def __init__(self, system, models):
        self.system = system
        self.models = models
        self.patterns = system.get('fallback_patterns', [])
        self.validation_rules = system.get('validation_rules', {})
        
        # Get best model info
        self.best_model_key = None
        self.best_model_config = None
        
        if models:
            # Use the first (best performing) loaded model
            self.best_model_key = list(models.keys())[0]
            self.best_model_config = models[self.best_model_key]
    
    def extract_movements(self, text):
        """Extract movements using LLM + patterns"""
        movements = []
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Method 1: Try LLM extraction
        if self.models and self.best_model_key:
            try:
                llm_movements = self.extract_with_llm(cleaned_text)
                movements.extend(llm_movements)
                
                # If LLM found good results, use them
                if len(llm_movements) >= 3:
                    return self.deduplicate_and_validate(movements)
            except Exception as e:
                st.warning(f"⚠️ LLM extraction failed: {str(e)[:50]}... Using patterns.")
        
        # Method 2: Use trained patterns (always run as backup)
        pattern_movements = self.extract_with_patterns(cleaned_text)
        movements.extend(pattern_movements)
        
        return self.deduplicate_and_validate(movements)
    
    def extract_with_llm(self, text):
        """Extract using the best LLM model"""
        movements = []
        
        # Split text into processable chunks
        chunks = self.split_text_intelligently(text)
        
        model_pipeline = self.best_model_config['pipeline']
        prompt_template = self.best_model_config['config']['prompt_template']
        
        for chunk in chunks[:4]:  # Process max 4 chunks
            try:
                # Create prompt using trained template
                prompt = prompt_template.format(text=chunk)
                
                # Generate response
                response = model_pipeline(
                    prompt,
                    max_new_tokens=self.best_model_config['config']['generation_config']['max_new_tokens'],
                    temperature=self.best_model_config['config']['generation_config']['temperature'],
                    do_sample=self.best_model_config['config']['generation_config']['do_sample'],
                    return_full_text=self.best_model_config['config']['generation_config']['return_full_text'],
                    pad_token_id=50256,
                    eos_token_id=50256
                )
                
                generated_text = response[0]['generated_text']
                
                # Parse using trained strategies
                chunk_movements = self.parse_llm_response(generated_text, chunk)
                movements.extend(chunk_movements)
                
            except Exception as e:
                continue  # Skip problematic chunks
        
        return movements
    
    def extract_with_patterns(self, text):
        """Extract using trained regex patterns"""
        movements = []
        
        for pattern, movement_type in self.patterns:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    person, company = self.extract_entities_from_groups(groups, movement_type, match.group(0))
                    
                    if person and company and self.is_valid_name(person):
                        movements.append({
                            'name': person.strip(),
                            'company': company.strip(),
                            'movement_type': movement_type,
                            'confidence': 0.9,
                            'method': 'trained_patterns',
                            'context': match.group(0)[:200]
                        })
            except Exception as e:
                continue  # Skip problematic patterns
        
        return movements
    
    def split_text_intelligently(self, text):
        """Split text into meaningful chunks for LLM processing"""
        # Look for article separators or paragraph breaks
        potential_chunks = []
        
        # Split by multiple newlines (article separators)
        sections = re.split(r'\n{2,}', text)
        
        for section in sections:
            section = section.strip()
            if len(section) > 100 and self.has_talent_indicators(section):
                # Further split long sections
                if len(section) > 800:
                    sentences = re.split(r'[.!?]\s+', section)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 600:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk and self.has_talent_indicators(current_chunk):
                                potential_chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    
                    if current_chunk and self.has_talent_indicators(current_chunk):
                        potential_chunks.append(current_chunk.strip())
                else:
                    potential_chunks.append(section)
        
        return potential_chunks[:5]  # Limit to 5 chunks for performance
    
    def has_talent_indicators(self, text):
        """Check if text contains talent movement indicators"""
        indicators = [
            'joins', 'hires', 'appoints', 'promotes', 'taps', 'named', 'picked',
            'launch', 'debut', 'founding', 'starting', 'preps', 'plots', 'eyes',
            'Capital', 'Management', 'Fund', 'CEO', 'CIO', 'COO', 'CFO', 'director'
        ]
        return any(indicator in text for indicator in indicators)
    
    def parse_llm_response(self, response, original_chunk):
        """Parse LLM response using multiple strategies"""
        movements = []
        
        # Strategy 1: Trained format "Name at Company"
        pattern1 = r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+at\s+([A-Z][A-Za-z\s]+?)(?:\s|$|\.)'
        matches1 = re.finditer(pattern1, response, re.IGNORECASE)
        for match in matches1:
            if self.is_valid_name(match.group(1)):
                movements.append({
                    'name': match.group(1).strip(),
                    'company': match.group(2).strip(),
                    'movement_type': 'movement',
                    'confidence': 0.85,
                    'method': 'llm_trained_format',
                    'context': original_chunk[:200]
                })
        
        # Strategy 2: Natural language patterns
        natural_patterns = [
            (r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:joins|joined)\s+([A-Z][A-Za-z\s]+)', 'hire'),
            (r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:starts|started|launches|founded)\s+([A-Z][A-Za-z\s]+)', 'launch'),
            (r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:promoted|appointed|named)\s+.*?([A-Z][A-Za-z\s]+)', 'promotion'),
        ]
        
        for pattern, movement_type in natural_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                if self.is_valid_name(match.group(1)):
                    movements.append({
                        'name': match.group(1).strip(),
                        'company': match.group(2).strip(),
                        'movement_type': movement_type,
                        'confidence': 0.8,
                        'method': 'llm_natural_language',
                        'context': original_chunk[:200]
                    })
        
        # Strategy 3: Look in original chunk if LLM response is unclear
        if len(movements) == 0:
            fallback_movements = self.extract_with_patterns(original_chunk)
            movements.extend(fallback_movements)
        
        return movements
    
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
    
    def extract_entities_from_groups(self, groups, movement_type, full_match):
        """Extract person and company from regex groups"""
        person, company = None, None
        
        if movement_type == "launch":
            if "'s" in full_match:
                person, company = groups[0], groups[1]
            else:
                person, company = groups[0], groups[1] if len(groups) > 1 else None
        elif movement_type in ["hire", "promotion"]:
            if "joins" in full_match:
                person, company = groups[0], groups[1]
            elif "hires" in full_match or "appoints" in full_match:
                company, person = groups[0], groups[1]
            elif "picked" in full_match:
                person = groups[0]
                company = self.find_company_in_context(full_match)
        
        return person, company
    
    def find_company_in_context(self, match_text):
        """Find company name in context"""
        company_patterns = [
            r'([A-Z][A-Za-z\s]{3,30}(?:Capital|Management|Fund|Group|Partners|Treasury))',
            r'([A-Z][A-Za-z\s]{5,25}(?:LLC|Inc|Corp))'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, match_text)
            if matches:
                return matches[0]
        
        return "Unknown Company"
    
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
    st.title("🤖 AI-Trained Talent Movement Tracker")
    st.markdown("**Powered by your trained LLM models + pattern fallbacks**")
    
    # Load trained system
    system = load_trained_system()
    if not system:
        st.stop()
    
    # Load models
    models = load_llm_models(system)
    
    # Sidebar with info
    st.sidebar.header("🧠 System Information")
    
    training_stats = system.get('performance_metrics', {})
    st.sidebar.info(f"""
    **Training Platform:** {system.get('platform', 'Unknown')}  
    **Training Date:** {system.get('training_date', 'Unknown')[:10]}  
    **Models Trained:** {len(system.get('models_trained', []))}  
    **Quality Score:** {training_stats.get('avg_quality_score', 0):.1%}
    """)
    
    if models:
        model_info = list(models.values())[0]
        st.sidebar.success(f"🤖 **Active Model:** {model_info['model_name']}")
        
        performance = model_info['config']['performance']
        st.sidebar.metric("Model Quality", f"{performance['quality']:.1%}")
        st.sidebar.metric("Avg Extractions", f"{performance['extraction_count']}")
    else:
        st.sidebar.warning("⚠️ No LLM models loaded")
        st.sidebar.info("Using trained patterns only")
    
    if system.get('best_combinations'):
        st.sidebar.subheader("🏆 Training Results")
        for i, combo in enumerate(system['best_combinations'][:3], 1):
            st.sidebar.write(f"{i}. **{combo['model']}** ({combo['quality']:.1%})")
    
    # File upload
    st.header("📁 Upload Newsletter Files")
    uploaded_files = st.file_uploader(
        "Upload newsletter .txt files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload financial newsletter files for AI-powered talent extraction"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} files uploaded")
        
        # Extraction options
        col1, col2 = st.columns(2)
        with col1:
            use_llm = st.checkbox("🤖 Use LLM models", value=bool(models), disabled=not models)
        with col2:
            use_patterns = st.checkbox("🔍 Use pattern fallbacks", value=True)
        
        if st.button("🚀 Extract Talent Movements", type="primary"):
            
            # Initialize extractor
            extractor = EnhancedExtractor(system, models if use_llm else {})
            
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
        
        # Add analysis columns safely
        context_col = df['context'].fillna('').astype(str) if 'context' in df.columns else pd.Series([''] * len(df))
        company_col = df['company'].fillna('').astype(str) if 'company' in df.columns else pd.Series([''] * len(df))
        
        df['asia_related'] = context_col.str.contains(
            'asia|china|singapore|hong kong|japan|korea|india|apac', 
            case=False, na=False
        )
        df['senior_level'] = context_col.str.contains(
            'ceo|cio|coo|cfo|chief|president|director|founder', 
            case=False, na=False
        )
        df['financial_firm'] = company_col.str.contains(
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
        st.subheader("🔧 Extraction Methods Used")
        method_counts = df['method'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(method_counts)
        with col2:
            for method, count in method_counts.items():
                pct = (count / len(df)) * 100
                st.write(f"**{method.replace('_', ' ').title()}:** {count} ({pct:.1f}%)")
        
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
        
        # Show individual movements
        st.subheader("👥 Individual Movements")
        
        for i, row in df.iterrows():
            with st.expander(f"👤 {row['name']} → {row['company']} ({row['movement_type']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Name:**", row['name'])
                    st.write("**Company:**", row['company'])
                
                with col2:
                    st.write("**Movement:**", row['movement_type'].title())
                    st.write("**Method:**", row['method'].replace('_', ' ').title())
                
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
                
                # Show context safely
                context_text = row.get('context', 'No context available')
                if context_text:
                    st.text_area("Context", context_text[:300], height=80, disabled=True, key=f"context_{i}")
        
        # Download section
        st.header("💾 Download Results")
        
        # Prepare export
        export_df = prepare_export_data(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"ai_talent_movements_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel with multiple sheets
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Movements', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total', 'LLM Extracted', 'Pattern Extracted', 'Asia-Related', 'Senior-Level'],
                    'Count': [
                        len(df),
                        len(df[df['method'].str.contains('llm', na=False)]),
                        len(df[df['method'].str.contains('pattern', na=False)]),
                        len(df[df['asia_related']]),
                        len(df[df['senior_level']])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"ai_talent_movements_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def prepare_export_data(df):
    """Prepare data for export"""
    export_df = pd.DataFrame()
    
    export_df['Full Name'] = df['name']
    export_df['Company'] = df['company']
    export_df['Movement Type'] = df['movement_type'].str.title()
    export_df['Extraction Method'] = df['method'].str.replace('_', ' ').str.title()
    export_df['Confidence Score'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    export_df['Asia Related'] = df['asia_related'].map({True: 'Yes', False: 'No'})
    export_df['Senior Level'] = df['senior_level'].map({True: 'Yes', False: 'No'})
    export_df['Financial Firm'] = df['financial_firm'].map({True: 'Yes', False: 'No'})
    export_df['Date Extracted'] = datetime.now().strftime('%Y-%m-%d')
    export_df['Source'] = 'AI-Trained System'
    
    return export_df

if __name__ == "__main__":
    main()
