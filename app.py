import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import uuid
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Asian Hedge Fund Talent Map",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'people' not in st.session_state:
    st.session_state.people = []
    # Add some sample Asian hedge fund data
    sample_people = [
        {
            "id": str(uuid.uuid4()),
            "name": "Li Wei Chen",
            "current_title": "Portfolio Manager",
            "current_company_name": "Hillhouse Capital",
            "location": "Hong Kong",
            "email": "li.chen@hillhouse.com",
            "linkedin_profile_url": "https://linkedin.com/in/liweichen",
            "phone": "+852-1234-5678",
            "education": "Harvard Business School, Tsinghua University",
            "expertise": "Technology, Healthcare",
            "aum_managed": "2.5B USD",
            "strategy": "Long-only Growth Equity"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Akira Tanaka",
            "current_title": "Chief Investment Officer",
            "current_company_name": "Millennium Partners Asia",
            "location": "Singapore",
            "email": "a.tanaka@millennium.com",
            "linkedin_profile_url": "https://linkedin.com/in/akiratanaka",
            "phone": "+65-9876-5432",
            "education": "Tokyo University, Wharton",
            "expertise": "Quantitative Trading, Fixed Income",
            "aum_managed": "1.8B USD",
            "strategy": "Multi-Strategy Quantitative"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Sarah Kim",
            "current_title": "Head of Research",
            "current_company_name": "Citadel Asia",
            "location": "Seoul",
            "email": "s.kim@citadel.com",
            "linkedin_profile_url": "https://linkedin.com/in/sarahkim",
            "phone": "+82-10-1234-5678",
            "education": "Seoul National University, MIT Sloan",
            "expertise": "Equity Research, ESG",
            "aum_managed": "800M USD",
            "strategy": "Equity Long/Short"
        }
    ]
    st.session_state.people.extend(sample_people)

if 'firms' not in st.session_state:
    st.session_state.firms = []
    # Add sample Asian hedge fund firms
    sample_firms = [
        {
            "id": str(uuid.uuid4()),
            "name": "Hillhouse Capital",
            "location": "Hong Kong",
            "headquarters": "Beijing, China",
            "aum": "60B USD",
            "founded": 2005,
            "strategy": "Long-only, Growth Equity",
            "website": "https://hillhousecap.com",
            "description": "Asia's largest hedge fund focusing on technology and healthcare investments"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Millennium Partners Asia",
            "location": "Singapore",
            "headquarters": "New York, USA",
            "aum": "35B USD",
            "founded": 1989,
            "strategy": "Multi-strategy, Quantitative",
            "website": "https://millennium.com",
            "description": "Global hedge fund with significant Asian operations"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Citadel Asia",
            "location": "Hong Kong",
            "headquarters": "Chicago, USA",
            "aum": "45B USD",
            "founded": 1990,
            "strategy": "Multi-strategy, Market Making",
            "website": "https://citadel.com",
            "description": "Leading global hedge fund with growing Asian presence"
        }
    ]
    st.session_state.firms.extend(sample_firms)

if 'employments' not in st.session_state:
    st.session_state.employments = []
    # Add sample employment history
    for person in st.session_state.people:
        current_firm = next((f for f in st.session_state.firms if f['name'] == person['current_company_name']), None)
        if current_firm:
            st.session_state.employments.append({
                "id": str(uuid.uuid4()),
                "person_id": person['id'],
                "company_name": person['current_company_name'],
                "title": person['current_title'],
                "start_date": date(2020, 1, 1),
                "end_date": None,
                "location": person['location'],
                "strategy": person.get('strategy', 'Unknown')
            })

if 'all_extractions' not in st.session_state:
    st.session_state.all_extractions = []

if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'firms'  # 'firms', 'people', 'person_details', 'firm_details'

if 'selected_person_id' not in st.session_state:
    st.session_state.selected_person_id = None

if 'selected_firm_id' not in st.session_state:
    st.session_state.selected_firm_id = None

if 'show_add_person_modal' not in st.session_state:
    st.session_state.show_add_person_modal = False

if 'show_add_firm_modal' not in st.session_state:
    st.session_state.show_add_firm_modal = False

# --- Setup Gemini AI ---
@st.cache_resource
def setup_gemini(api_key, model_name='gemini-2.5-flash'):
    """Setup Gemini AI model"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error setting up Gemini: {e}")
        return None

def preprocess_newsletter(raw_text):
    """Clean and preprocess newsletter text for better AI extraction"""
    import re
    
    # Convert to string if bytes
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode('utf-8', errors='ignore')
    
    text = raw_text
    
    # Remove email headers (From:, Sent:, To:, Subject:)
    text = re.sub(r'^From:.*?Subject:.*?\n', '', text, flags=re.MULTILINE | re.DOTALL)
    
    # Remove long URLs and email links
    text = re.sub(r'https?://[^\s<>"]+', '', text)
    text = re.sub(r'mailto:[^\s<>"]+', '', text)
    
    # Remove HTML-like tags and formatting
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[^;]+;', ' ', text)  # HTML entities
    
    # Remove JP Morgan specific boilerplate
    boilerplate_patterns = [
        r'This section contains materials produced by third parties.*?for the results obtained from your use of such information\.',
        r'© 202\d JPMorgan Chase.*?(?=\n|$)',
        r'JPMorgan Chase Bank.*?FDIC insured\.',
        r'Important Reminder: JPMorgan Chase will never send emails.*?(?=\n|$)',
        r'This message is confidential.*?(?=\n|$)',
        r'Unsubscribe.*?(?=\n|$)',
        r'Privacy Policy.*?(?=\n|$)',
        r'J\.P\. Morgan Corporate & Investment Bank Marketing.*?(?=\n|$)',
        r'CAPITAL ADVISORY GROUP.*?(?=\n|$)',
        r'Hedge Fund News.*?(?=\n|$)',
        r'SEE ALL ARTICLES.*?(?=\n|$)',
        r'jpmorgan\.com.*?(?=\n|$)'
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove excessive whitespace and line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple line breaks to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    
    # Remove lines that are just separators or formatting
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, separators, and formatting junk
        if (len(line) > 10 and 
            not line.startswith('_') and 
            not line.startswith('=') and
            not line.startswith('|') and
            not re.match(r'^[^a-zA-Z]*$', line) and  # Skip lines with no letters
            'unsubscribe' not in line.lower() and
            'privacy policy' not in line.lower() and
            'jpmorgan' not in line.lower() and
            len(line.split()) > 2):  # Need at least 3 words
            cleaned_lines.append(line)
    
    # Join back and clean up
    text = '\n'.join(cleaned_lines)
    
    # Extract main content sections - look for news items
    # JP Morgan newsletters typically have "Source:" pattern
    news_items = []
    current_item = ""
    
    for line in text.split('\n'):
        line = line.strip()
        if line:
            if line.startswith('Source:'):
                if current_item:
                    news_items.append(current_item.strip())
                current_item = ""
            else:
                current_item += " " + line
    
    # Add the last item
    if current_item:
        news_items.append(current_item.strip())
    
    # If we found structured news items, use those
    if news_items and len(news_items) > 2:
        text = '\n\n'.join(news_items)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    return text

def extract_talent_with_batching(newsletter_text, model, preprocessing_mode="smart", debug_mode=False, chunk_size=15000, overlap_size=1000):
    """Extract with custom batching parameters"""
    
    if preprocessing_mode == "smart":
        cleaned_text = preprocess_newsletter(newsletter_text)
    elif preprocessing_mode == "raw":
        cleaned_text = newsletter_text.replace('\n', ' ').strip()
    else:
        cleaned_text = newsletter_text
    
    # Always use batching when this function is called
    return extract_with_batching(cleaned_text, model, chunk_size, overlap_size, debug_mode)

def extract_talent(newsletter_text, model, preprocessing_mode="smart", debug_mode=False):
    """Extract hedge fund talent movements using Gemini AI with batching for large texts"""
    
    if preprocessing_mode == "smart":
        cleaned_text = preprocess_newsletter(newsletter_text)
    elif preprocessing_mode == "raw":
        cleaned_text = newsletter_text.replace('\n', ' ').strip()
    else:
        cleaned_text = newsletter_text
    
    # Check text length and decide on batching
    max_chunk_size = 15000  # characters per chunk
    overlap_size = 1000     # overlap between chunks to avoid missing cross-boundary names
    
    if len(cleaned_text) <= max_chunk_size:
        # Single chunk processing
        return extract_single_chunk(cleaned_text, model, debug_mode)
    else:
        # Multi-chunk processing with batching
        return extract_with_batching(cleaned_text, model, max_chunk_size, overlap_size, debug_mode)

def extract_single_chunk(text, model, debug_mode=False):
    """Extract from a single chunk of text"""
    
    prompt = f"""
You are extracting financial talent movements. Extract EVERY person mentioned in professional contexts.

NEWSLETTER TEXT ({len(text):,} characters):
{text}

Extract in JSON format:
{{
  "extractions": [
    {{
      "name": "First Last",
      "company": "Current/New Company",
      "previous_company": "Previous Company (if mentioned)",
      "movement_type": "launch|hire|promotion|departure|appointment|partnership",
      "title": "Position",
      "location": "Location",
      "strategy": "Strategy/Focus",
      "context": "What happened"
    }}
  ]
}}

CAPTURE ALL PATTERNS:
- "X's new company Y" = X launches Y
- "X joins Y" = X hired by Y  
- "X promoted to" = X promotion
- "X picked for position" = X hired/promoted
- "X would lead" = X appointment
- "X and Y as VPs" = TWO separate hire entries
- "ex-Goldman X" = X has previous_company Goldman
- "former DB executive X" = X has previous_company Deutsche Bank

Find EVERY person with first and last name. Be exhaustive.
"""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        if debug_mode:
            st.text_area(f"AI Response (Single Chunk - {len(text):,} chars):", response_text, height=200)
        
        return parse_ai_response(response_text)
    except Exception as e:
        st.error(f"Single chunk extraction error: {e}")
        return []

def extract_with_batching(text, model, chunk_size, overlap, debug_mode=False):
    """Extract from large text using batching with overlap"""
    
    chunks = []
    start = 0
    
    # Create overlapping chunks
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        # Try to break at sentence boundary near the end
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > len(chunk) * 0.8:  # Only break if it's near the end
                chunk = chunk[:break_point + 1]
                end = start + len(chunk)
        
        chunks.append({
            'text': chunk,
            'start': start,
            'end': end,
            'size': len(chunk)
        })
        
        start = end - overlap  # Overlap for continuity
        if start >= len(text):
            break
    
    st.info(f"📊 Processing {len(chunks)} chunks of ~{chunk_size:,} characters each with {overlap:,} char overlap")
    
    all_extractions = []
    seen_names = set()  # Deduplicate across chunks
    
    # Process each chunk
    progress_container = st.empty()
    
    for i, chunk_info in enumerate(chunks):
        progress_container.write(f"🔄 Processing chunk {i+1}/{len(chunks)} ({chunk_info['size']:,} chars)...")
        
        chunk_prompt = f"""
Extract financial talent movements from this newsletter chunk.

CHUNK {i+1}/{len(chunks)} ({chunk_info['size']:,} characters):
{chunk_info['text']}

Extract in JSON format:
{{
  "extractions": [
    {{
      "name": "First Last",
      "company": "Current/New Company", 
      "previous_company": "Previous Company (if mentioned)",
      "movement_type": "launch|hire|promotion|departure|appointment|partnership",
      "title": "Position",
      "location": "Location", 
      "strategy": "Strategy/Focus",
      "context": "What happened"
    }}
  ]
}}

Find EVERY person mentioned in career contexts. Include:
- Fund launches, hires, promotions, departures
- "X's company", "X joins Y", "X promoted", "former Y executive X"
- Portfolio managers, CIOs, VPs, directors, analysts
- Look for first and last names only

Be thorough - extract everyone mentioned professionally.
"""
        
        try:
            response = model.generate_content(chunk_prompt)
            response_text = response.text
            
            if debug_mode:
                st.text_area(f"AI Response Chunk {i+1}:", response_text, height=150)
            
            chunk_extractions = parse_ai_response(response_text)
            
            # Deduplicate based on name + company combination
            new_extractions = 0
            for extraction in chunk_extractions:
                name = extraction.get('name', '').strip()
                company = extraction.get('company', '').strip()
                
                if name and company:
                    key = f"{name.lower()}|{company.lower()}"
                    if key not in seen_names:
                        seen_names.add(key)
                        all_extractions.append(extraction)
                        new_extractions += 1
            
            progress_container.success(f"✅ Chunk {i+1}/{len(chunks)}: Found {len(chunk_extractions)} movements ({new_extractions} new, {len(all_extractions)} total)")
            
        except Exception as e:
            progress_container.error(f"❌ Error processing chunk {i+1}: {e}")
            continue
    
    progress_container.success(f"🎯 **Batching Complete**: Found {len(all_extractions)} total unique movements from {len(chunks)} chunks")
    return all_extractions

def parse_ai_response(response_text):
    """Parse AI response and extract JSON"""
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_text = response_text[json_start:json_end]
        try:
            result = json.loads(json_text)
            extractions = result.get('extractions', [])
            
            # Filter out invalid entries
            valid_extractions = []
            for ext in extractions:
                if ext.get('name') and ext.get('company'):
                    valid_extractions.append(ext)
            
            return valid_extractions
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {e}")
            st.text_area("Invalid JSON response:", json_text, height=200)
            return []
    else:
        st.error("No valid JSON found in AI response")
        st.text_area("Full AI response:", response_text, height=200) 
        return []

# --- Helper Functions ---
def get_person_by_id(person_id):
    return next((p for p in st.session_state.people if p['id'] == person_id), None)

def get_firm_by_id(firm_id):
    return next((f for f in st.session_state.firms if f['id'] == firm_id), None)

def get_firm_by_name(firm_name):
    return next((f for f in st.session_state.firms if f['name'] == firm_name), None)

def get_people_by_firm(firm_name):
    return [p for p in st.session_state.people if p['current_company_name'] == firm_name]

def get_employments_by_person_id(person_id):
    return [e for e in st.session_state.employments if e['person_id'] == person_id]

def calculate_overlap_years(start1, end1, start2, end2):
    today = date.today()
    period1_end = end1 if end1 is not None else today
    period2_end = end2 if end2 is not None else today
    
    latest_start = max(start1, start2)
    earliest_end = min(period1_end, period2_end)
    
    overlap_days = (earliest_end - latest_start).days
    if overlap_days <= 0:
        return 0.0
    return round(overlap_days / 365.25, 2)

# --- Navigation Functions ---
def go_to_firms():
    st.session_state.current_view = 'firms'
    st.session_state.selected_firm_id = None

def go_to_people():
    st.session_state.current_view = 'people'
    st.session_state.selected_person_id = None

def go_to_person_details(person_id):
    st.session_state.selected_person_id = person_id
    st.session_state.current_view = 'person_details'

def go_to_firm_details(firm_id):
    st.session_state.selected_firm_id = firm_id
    st.session_state.current_view = 'firm_details'

# --- SIDEBAR: AI Talent Extractor ---
with st.sidebar:
    st.title("🤖 AI Talent Extractor")
    
    # Show improvements banner
    with st.container():
        st.success("""
        ✅ **MAJOR IMPROVEMENTS ADDED:**
        • **Full Text Processing** - No more 2K character limits!
        • **Smart Batching** - Handles large newsletters automatically  
        • **Overlap Protection** - Prevents missing names at chunk boundaries
        • **Deduplication** - Removes duplicate entries across chunks
        • **Better Employment History** - Tracks current + previous companies
        """)
    
    # API Key Setup
    st.subheader("🔑 Configuration")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API key loaded")
    except:
        api_key = st.text_input("Gemini API Key", type="password", help="Get from: https://makersuite.google.com/app/apikey")
    
    # Model Selection
    st.subheader("🤖 Model Selection")
    model_options = {
        "Gemini 2.5 Flash (BEST)": "gemini-2.5-flash",
        "Gemini 2.0 Flash": "gemini-2.0-flash", 
        "Gemini 1.5 Flash": "gemini-1.5-flash",
        "Gemini 1.5 Pro (Advanced)": "gemini-1.5-pro"
    }
    
    selected_model_name = st.selectbox(
        "Choose extraction model:",
        options=list(model_options.keys()),
        index=0,  # Default to 2.5 Flash
        help="2.5 Flash recommended for best extraction accuracy"
    )
    
    selected_model_id = model_options[selected_model_name]
    
    # Show model info
    if "2.5-flash" in selected_model_id:
        st.info("🌟 **Best choice**: Most accurate extraction, latest model")
    elif "2.0-flash" in selected_model_id:
        st.info("⚡ **Great choice**: Fast and accurate, good rate limits")
    elif "1.5-pro" in selected_model_id:
        st.warning("🧠 **Advanced**: Best reasoning but limited rate (2 RPM)")
    elif "1.5-flash" in selected_model_id:
        st.info("📊 **Standard**: Basic model, may miss some entries")
    
    if api_key:
        model = setup_gemini(api_key, selected_model_id)
        
        # Model Performance Tips
        with st.expander("📊 Model Performance Guide"):
            st.markdown("""
            **🏆 Gemini 2.5 Flash (Recommended)**
            - Most accurate extraction (15-20+ entries expected)
            - Best at following complex instructions
            - Superior pattern recognition
            - Rate limit: 10 RPM / 250K TPM
            
            **⚡ Gemini 2.0 Flash**  
            - Very good extraction (12-18 entries expected)
            - Fast processing
            - Good instruction following
            - Rate limit: 15 RPM / 1M TPM
            
            **🧠 Gemini 1.5 Pro**
            - Best reasoning for ambiguous cases
            - Excellent for complex newsletters
            - **Warning**: Only 2 RPM (very limited!)
            - Rate limit: 2 RPM / 32K TPM
            
            **📊 Gemini 1.5 Flash (Current)**
            - Basic extraction (6-12 entries typical)
            - May miss some subtle movements
            - Fastest processing
            - Rate limit: 15 RPM / 1M TPM
            
            💡 **All models are FREE on the free tier!**
            """)
        
        # Test different models button
        if st.button("🔬 Compare Models on Sample", use_container_width=True):
            sample_text = """
            Harrison Balistreri's Inevitable Capital Management will trade l/s strat.
            Davidson Kempner eyes European strat for l/s equity co-head New standalone strategy would be led by Vince Ortiz.
            Ex-Marshall Wace healthcare PM plots HF debut Robin Boldt to debut ROCK2 Capital in London.
            Tennessee Treasury promotes PE director to deputy CIO Daniel Crews picked for position; senior PM Grant Leslie to lead PE.
            GS vet teams up with ex-Paloma co-CIO on quant launch Sarah Gray joins Neil Chriss on forming Edge Peak.
            Options Group strengthens with two new VPs Louis Couronne and Macaire Chue as vice presidents.
            """
            
            st.write("**Testing all models on same sample:**")
            
            models_to_test = [
                ("Gemini 2.5 Flash", "gemini-2.5-flash"),
                ("Gemini 2.0 Flash", "gemini-2.0-flash"), 
                ("Gemini 1.5 Flash", "gemini-1.5-flash")
            ]
            
            for model_name, model_id in models_to_test:
                try:
                    test_model = setup_gemini(api_key, model_id)
                    if test_model:
                        with st.spinner(f"Testing {model_name}..."):
                            extractions = extract_talent(sample_text, test_model, "raw", False)
                            count = len(extractions) if extractions else 0
                            st.write(f"**{model_name}**: {count} movements extracted")
                except Exception as e:
                    st.write(f"**{model_name}**: Error - {str(e)}")
            
            st.info("💡 Higher extraction count = better model for your use case")
        
        st.markdown("---")
        # Preprocessing options
        st.subheader("📰 Extract from Newsletter")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            preprocessing_mode = st.radio(
                "Text processing mode:",
                ["🧹 Smart Clean (Recommended)", "📄 Raw Text (Debug)", "✂️ Manual Clean"],
                help="Try 'Raw Text' if Smart Clean misses entries"
            )
        with col2:
            debug_mode = st.checkbox("🐛 Show AI responses", help="Shows raw AI output for debugging")
            
        # Batching controls
        st.subheader("⚡ Batching Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.number_input("Chunk size (chars)", value=15000, min_value=5000, max_value=50000, step=5000, 
                                       help="Larger chunks = fewer API calls but may hit limits")
        with col2:
            overlap_size = st.number_input("Overlap size (chars)", value=1000, min_value=500, max_value=5000, step=500,
                                         help="Overlap prevents missing names at boundaries")
        with col3:
            auto_batch = st.checkbox("Auto-batch large files", value=True, help="Automatically split large newsletters")
        
        input_method = st.radio("Input method:", ["📝 Text", "📁 File"])
        
        newsletter_text = ""
        if input_method == "📝 Text":
            newsletter_text = st.text_area("Newsletter content:", height=200, placeholder="Paste Asian hedge fund newsletter content...")
            if newsletter_text:
                char_count = len(newsletter_text)
                st.info(f"📊 Text length: {char_count:,} characters")
                if char_count > 15000:
                    st.warning(f"⚠️ Large text detected! Recommend using batching (will process in ~{max(1, char_count//15000)} chunks)")
        else:
            uploaded_file = st.file_uploader("Upload file:", type=['txt'])
            if uploaded_file:
                try:
                    raw_data = uploaded_file.read()
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            newsletter_text = raw_data.decode(encoding)
                            char_count = len(newsletter_text)
                            st.success(f"✅ File loaded: {char_count:,} characters")
                            if char_count > 15000:
                                estimated_movements = max(15, (char_count // 1000) * 2)  # Rough estimate
                                st.warning(f"⚠️ Large file detected! Recommend using batching (will process in ~{max(1, char_count//15000)} chunks)")
                                st.info(f"🎯 **Expected extractions:** ~{estimated_movements} movements based on file length")
                            break
                        except UnicodeDecodeError:
                            continue
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("🚀 Extract Talent", use_container_width=True):
            if newsletter_text.strip() and model:
                with st.spinner("🤖 Analyzing..."):
                    # Debug: Show preprocessing results and actual lengths
                    with st.expander("🔍 Debug Info - Text Processing Analysis"):
                        if preprocessing_mode.startswith("🧹"):
                            cleaned = preprocess_newsletter(newsletter_text)
                            st.write(f"**Mode:** Smart Clean")
                        elif preprocessing_mode.startswith("📄"):
                            cleaned = newsletter_text.replace('\n', ' ').strip()
                            st.write(f"**Mode:** Raw Text")
                        else:
                            cleaned = newsletter_text
                            st.write(f"**Mode:** Manual Clean")
                            
                        st.write(f"**Original length:** {len(newsletter_text):,} characters")
                        st.write(f"**After preprocessing:** {len(cleaned):,} characters")
                        
                        # Show batching info
                        if auto_batch and len(cleaned) > chunk_size:
                            num_chunks = max(1, (len(cleaned) - overlap_size) // (chunk_size - overlap_size))
                            st.warning(f"📊 **Large file detected!** Will process in {num_chunks} chunks of ~{chunk_size:,} chars each")
                            st.write(f"**Chunk size:** {chunk_size:,} chars")
                            st.write(f"**Overlap:** {overlap_size:,} chars")
                        else:
                            st.success(f"✅ **Single chunk processing** - Sending full {len(cleaned):,} characters to AI")
                        
                        st.write(f"**Preview (first 2000 chars of {len(cleaned):,} total):**")
                        st.text_area("Text preview:", cleaned[:2000], height=200)
                        
                        # Quick manual check
                        manual_check = [
                            "Harrison Balistreri", "Vince Ortiz", "Robin Boldt", 
                            "Daniel Crews", "Sarah Gray", "Neil Chriss",
                            "Louis Couronne", "Macaire Chue", "Grant Leslie",
                            "Jo-Wen Lin", "Michael Furla", "Alberto Cozzini"
                        ]
                        
                        st.write("**Quick check - Are these names in the full text?**")
                        found_count = 0
                        for name in manual_check:
                            found = name.lower() in cleaned.lower()
                            status = "✅" if found else "❌"
                            if found:
                                found_count += 1
                            st.write(f"{status} {name}")
                        
                        st.write(f"**Found {found_count}/{len(manual_check)} target names in text**")
                    
                    # Pass batching parameters
                    if auto_batch:
                        extractions = extract_talent_with_batching(newsletter_text, model, preprocessing_mode.split()[0].lower(), debug_mode, chunk_size, overlap_size)
                    else:
                        extractions = extract_talent(newsletter_text, model, preprocessing_mode.split()[0].lower(), debug_mode)
                    if extractions:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for ext in extractions:
                            ext['timestamp'] = timestamp
                        st.session_state.all_extractions.extend(extractions)
                        st.success(f"✅ Found {len(extractions)} movements!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No movements found")
            else:
                st.error("❌ Please provide content and API key")
        
        if st.button("🧪 Test Sample", use_container_width=True):
            sample_text = """
            Harrison Balistreri's Inevitable Capital Management will trade l/s strat.
            Davidson Kempner eyes European strat for l/s equity co-head New standalone strategy would be led by Vince Ortiz.
            Ex-Marshall Wace healthcare PM plots HF debut Robin Boldt to debut ROCK2 Capital in London.
            Tennessee Treasury promotes PE director to deputy CIO Daniel Crews picked for position; senior PM Grant Leslie to lead PE.
            GS vet teams up with ex-Paloma co-CIO on quant launch Sarah Gray joins Neil Chriss on forming Edge Peak.
            Options Group strengthens with two new VPs Louis Couronne and Macaire Chue as vice presidents.
            Ex-DB distressed head, Fortress Euro credit co-head eye debut Gavin Colquhoun and Rahul Ahuja working on as-yet-unnamed firm.
            Credit Suisse alum's Arini plots multi-PM HF launch Hamza Lemssouguer's London firm to debut Arini Diversified Alpha.
            Man AHL, Theorema alum eyes quant macro launch Alberto Cozzini readies strategy under Polymathique moniker.
            """
            if model:
                st.write(f"**Testing with sample ({len(sample_text):,} characters):**")
                with st.spinner("Testing extraction..."):
                    extractions = extract_talent(sample_text, model, "raw", False)
                    if extractions:
                        st.success(f"✅ Found {len(extractions)} movements from sample!")
                        st.write("**Expected: 13+ movements** (Harrison, Vince, Robin, Daniel, Grant, Sarah, Neil, Louis, Macaire, Gavin, Rahul, Hamza, Alberto)")
                        
                        if len(extractions) < 10:
                            st.warning("⚠️ Low extraction count - AI may not be following instructions properly")
                        
                        with st.expander("Sample Results"):
                            for ext in extractions:
                                name = ext.get('name', 'Unknown')
                                company = ext.get('company', 'Unknown')
                                movement = ext.get('movement_type', 'Unknown')
                                st.write(f"• **{name}** → {company} ({movement})")
                    else:
                        st.error("❌ No movements found in sample - check AI model and prompt")
        
        # Batching test
        if st.button("⚡ Test Batching (Large Sample)", use_container_width=True):
            # Create a large sample by repeating content
            base_sample = """
            Harrison Balistreri's Inevitable Capital Management will trade l/s strat. Davidson Kempner eyes European strat led by Vince Ortiz.
            Ex-Marshall Wace healthcare PM Robin Boldt to debut ROCK2 Capital in London. Tennessee Treasury promotes Daniel Crews to deputy CIO.
            GS vet Sarah Gray joins Neil Chriss on forming Edge Peak. Options Group hires Louis Couronne and Macaire Chue as VPs.
            """
            
            # Repeat with variations to create large sample
            large_sample = ""
            for i in range(10):  # Create ~20k characters
                large_sample += f"\n--- Newsletter Section {i+1} ---\n" + base_sample
                
            if model:
                st.write(f"**Testing batching with large sample ({len(large_sample):,} characters):**")
                with st.spinner("Testing batched extraction..."):
                    extractions = extract_talent_with_batching(large_sample, model, "raw", False, chunk_size, overlap_size)
                    if extractions:
                        st.success(f"✅ Batching found {len(extractions)} unique movements!")
                        st.write("**Expected: 6+ unique people** (should deduplicate across chunks)")
                        
                        with st.expander("Batching Results"):
                            for ext in extractions:
                                name = ext.get('name', 'Unknown')
                                company = ext.get('company', 'Unknown')
                                st.write(f"• **{name}** → {company}")
                    else:
                        st.error("❌ Batching failed")
        
        # Simple name extraction test
        if st.button("🔍 Simple Name Test", use_container_width=True):
            if newsletter_text.strip() and model:
                st.write("**Step 1: Can AI find any names at all?**")
                
                simple_prompt = f"""
Find ALL people's names mentioned in this financial newsletter:

{newsletter_text[:2000]}

Just list the names you find, one per line. Look for:
- First and Last names of people
- Anyone mentioned in professional contexts
- Fund managers, executives, analysts, etc.

Example output:
John Smith
Mary Johnson
Robert Lee
"""
                
                try:
                    response = model.generate_content(simple_prompt)
                    names_found = response.text.strip().split('\n')
                    
                    st.write(f"**Found {len(names_found)} names:**")
                    for name in names_found[:20]:  # Show first 20
                        if name.strip():
                            st.write(f"• {name.strip()}")
                    
                    if len(names_found) < 5:
                        st.error("❌ AI is having trouble finding names - this explains low extraction count")
                    else:
                        st.success(f"✅ AI can find names! Issue is likely with JSON structuring")
                        
                except Exception as e:
                    st.error(f"Name test error: {e}")
        
        # Manual paste test
        st.markdown("### 🧪 Manual Test")
        manual_text = st.text_area(
            "Paste a small section to test:",
            placeholder="Harrison Balistreri's Inevitable Capital Management will trade l/s strat. Davidson Kempner eyes European strat led by Vince Ortiz.",
            height=100
        )
        
        if st.button("Test Manual Text", use_container_width=True) and manual_text:
            if model:
                st.write(f"**Testing manual text ({len(manual_text):,} characters):**")
                extractions = extract_talent(manual_text, model, "raw", False)
                st.write(f"**Found {len(extractions)} from manual text:**")
                for ext in extractions:
                    name = ext.get('name', 'Unknown')
                    company = ext.get('company', 'Unknown')
                    st.write(f"• {name} → {company}")
        
        # Force extraction test
        if st.button("🎯 Force Extract Specific Names", use_container_width=True):
            if newsletter_text.strip() and model:
                st.write("**Testing if AI can find these specific people from your newsletter:**")
                
                target_names = [
                    ("Harrison Balistreri", "Inevitable Capital Management"),
                    ("Vince Ortiz", "Davidson Kempner"), 
                    ("Daniel Crews", "Tennessee Treasury"),
                    ("Grant Leslie", "Tennessee Treasury"),
                    ("Sarah Gray", "Edge Peak"),
                    ("Neil Chriss", "Edge Peak")
                ]
                
                # Use first chunk of text for targeted search
                search_text = newsletter_text[:20000]  # Use more text than before
                
                forced_prompt = f"""
NEWSLETTER TEXT ({len(search_text):,} characters):
{search_text}

Find these SPECIFIC people and their movements from the text above:
- Harrison Balistreri and Inevitable Capital Management
- Vince Ortiz and Davidson Kempner 
- Daniel Crews and Tennessee Treasury
- Grant Leslie and Tennessee Treasury
- Sarah Gray and Edge Peak
- Neil Chriss and Edge Peak

Return JSON format:
{{"extractions": [...]}}
"""
                
                try:
                    response = model.generate_content(forced_prompt)
                    response_text = response.text
                    
                    forced_extractions = parse_ai_response(response_text)
                    
                    st.write(f"**Forced extraction found:** {len(forced_extractions)} people")
                    for ext in forced_extractions:
                        name = ext.get('name', 'Unknown')
                        company = ext.get('company', 'Unknown')
                        st.write(f"• {name} → {company}")
                        
                    if len(forced_extractions) < 3:
                        st.error("❌ AI struggling to find even specific names - text preprocessing may be removing content")
                    else:
                        st.success("✅ AI can find specific names when prompted directly")
                        
                except Exception as e:
                    st.error(f"Forced extraction error: {e}")
        
        # Export button
        if st.session_state.all_extractions:
            if st.button("📥 Export Extractions as CSV", use_container_width=True):
                df = pd.DataFrame(st.session_state.all_extractions)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"hedge_fund_extractions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Recent Extractions
    if st.session_state.all_extractions:
        st.markdown("---")
        st.subheader("📊 All Extractions")
        st.metric("Total Extracted", len(st.session_state.all_extractions))
        
        # Show all extractions with status
        for i, ext in enumerate(st.session_state.all_extractions):
            # Safe handling for None values
            ext_name = ext.get('name', '') or ''
            ext_company = ext.get('company', '') or ''
            
            if not ext_name:  # Skip entries with no name
                continue
                
            # Check if already added
            is_added = any(p['name'].lower() == ext_name.lower() for p in st.session_state.people if p.get('name'))
            status = "✅ Added" if is_added else "⏳ Pending"
            
            with st.expander(f"{status} | {ext_name} → {ext_company}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Type:** {ext.get('movement_type', 'Unknown')}")
                    if ext.get('title'):
                        st.write(f"**Title:** {ext['title']}")
                    if ext.get('location'):
                        st.write(f"**Location:** {ext['location']}")
                    if ext.get('strategy'):
                        st.write(f"**Strategy:** {ext['strategy']}")
                    if ext.get('previous_company'):
                        st.write(f"**Previous:** {ext['previous_company']}")
                    if ext.get('context'):
                        st.write(f"**Context:** {ext['context']}")
                    st.write(f"**Extracted:** {ext.get('timestamp', 'Unknown')}")
                
                with col2:
                    if not is_added:
                        if st.button(f"➕ Add", key=f"add_{i}_{ext.get('timestamp', '')}"):
                            # Add to people and firms
                            new_person_id = str(uuid.uuid4())
                            new_person = {
                                "id": new_person_id,
                                "name": ext_name,
                                "current_title": ext.get('title', 'Unknown'),
                                "current_company_name": ext_company,
                                "location": ext.get('location', 'Unknown'),
                                "email": "",
                                "linkedin_profile_url": "",
                                "phone": "",
                                "education": "",
                                "expertise": ext.get('strategy', ''),
                                "aum_managed": "",
                                "strategy": ext.get('strategy', 'Unknown')
                            }
                            st.session_state.people.append(new_person)
                            
                            # Add current company if doesn't exist
                            if ext_company and not get_firm_by_name(ext_company):
                                new_firm_id = str(uuid.uuid4())
                                st.session_state.firms.append({
                                    "id": new_firm_id,
                                    "name": ext_company,
                                    "location": ext.get('location', 'Unknown'),
                                    "headquarters": "Unknown",
                                    "aum": "Unknown",
                                    "founded": None,
                                    "strategy": ext.get('strategy', 'Hedge Fund'),
                                    "website": "",
                                    "description": f"Financial firm - {ext.get('strategy', 'Multi-strategy')}"
                                })
                            
                            # Add previous company if mentioned
                            prev_company = ext.get('previous_company')
                            if prev_company and not get_firm_by_name(prev_company):
                                prev_firm_id = str(uuid.uuid4())
                                st.session_state.firms.append({
                                    "id": prev_firm_id,
                                    "name": prev_company,
                                    "location": "Unknown",
                                    "headquarters": "Unknown", 
                                    "aum": "Unknown",
                                    "founded": None,
                                    "strategy": "Financial Services",
                                    "website": "",
                                    "description": f"Previous employer of {ext_name}"
                                })
                            
                            # Add current employment
                            if ext_company:
                                st.session_state.employments.append({
                                    "id": str(uuid.uuid4()),
                                    "person_id": new_person_id,
                                    "company_name": ext_company,
                                    "title": ext.get('title', 'Unknown'),
                                    "start_date": date.today(),
                                    "end_date": None,
                                    "location": ext.get('location', 'Unknown'),
                                    "strategy": ext.get('strategy', 'Unknown')
                                })
                            
                            # Add previous employment if mentioned
                            if prev_company:
                                # Estimate previous employment dates (1-5 years ago)
                                end_date = date.today() - timedelta(days=30)  # Left 1 month ago
                                start_date = end_date - timedelta(days=365*3)  # 3 years duration
                                
                                st.session_state.employments.append({
                                    "id": str(uuid.uuid4()),
                                    "person_id": new_person_id,
                                    "company_name": prev_company,
                                    "title": "Previous Role",
                                    "start_date": start_date,
                                    "end_date": end_date,
                                    "location": "Unknown",
                                    "strategy": "Unknown"
                                })
                            
                            st.success(f"✅ Added {ext_name} with employment history!")
                            st.rerun()
                    else:
                        st.success("Already Added ✅")

# --- MAIN CONTENT AREA ---
st.title("🏢 Asian Hedge Fund Talent Map")
st.markdown("### Professional network mapping for Asia's hedge fund industry")

# Top Navigation
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

with col1:
    if st.button("🏢 Firms", use_container_width=True, type="primary" if st.session_state.current_view == 'firms' else "secondary"):
        go_to_firms()
        st.rerun()

with col2:
    if st.button("👥 People", use_container_width=True, type="primary" if st.session_state.current_view == 'people' else "secondary"):
        go_to_people()
        st.rerun()

with col3:
    if st.button("➕ Add Person", use_container_width=True):
        st.session_state.show_add_person_modal = True
        st.rerun()

with col4:
    if st.button("🏢➕ Add Firm", use_container_width=True):
        st.session_state.show_add_firm_modal = True
        st.rerun()

with col5:
    # Quick stats
    st.metric("Total People", len(st.session_state.people))
    
# --- ADD PERSON MODAL ---
if st.session_state.show_add_person_modal:
    st.markdown("---")
    st.subheader("➕ Add New Person to Network")
    
    with st.form("add_person_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", placeholder="Li Wei Chen")
            title = st.text_input("Current Title*", placeholder="Portfolio Manager")
            company = st.selectbox("Current Company*", 
                                 options=[""] + [f['name'] for f in st.session_state.firms],
                                 help="Select existing firm or add new firm first")
            location = st.selectbox("Location*", 
                                  options=["", "Hong Kong", "Singapore", "Tokyo", "Seoul", "Mumbai", "Shanghai", "Beijing", "Taipei", "Bangkok", "Jakarta"])
            
        with col2:
            email = st.text_input("Email", placeholder="li.chen@company.com")
            phone = st.text_input("Phone", placeholder="+852-1234-5678")
            linkedin = st.text_input("LinkedIn URL", placeholder="https://linkedin.com/in/username")
            education = st.text_input("Education", placeholder="Harvard, Tsinghua University")
        
        col3, col4 = st.columns(2)
        with col3:
            expertise = st.text_input("Expertise", placeholder="Technology, Healthcare")
            aum = st.text_input("AUM Managed", placeholder="2.5B USD")
        
        with col4:
            start_date = st.date_input("Start Date at Current Company", value=date.today())
            strategy = st.selectbox("Investment Strategy", 
                                  options=["", "Equity Long/Short", "Multi-Strategy", "Quantitative", "Macro", "Credit", "Healthcare", "Technology", "Event Driven", "Distressed", "Market Neutral", "Long-only", "Activist"])
        
        submitted = st.form_submit_button("Add Person")
        
        if submitted:
            if name and title and company and location:
                new_person_id = str(uuid.uuid4())
                st.session_state.people.append({
                    "id": new_person_id,
                    "name": name,
                    "current_title": title,
                    "current_company_name": company,
                    "location": location,
                    "email": email,
                    "linkedin_profile_url": linkedin,
                    "phone": phone,
                    "education": education,
                    "expertise": expertise,
                    "aum_managed": aum,
                    "strategy": strategy
                })
                
                # Add employment record
                st.session_state.employments.append({
                    "id": str(uuid.uuid4()),
                    "person_id": new_person_id,
                    "company_name": company,
                    "title": title,
                    "start_date": start_date,
                    "end_date": None,
                    "location": location,
                    "strategy": strategy
                })
                
                st.success(f"✅ Added {name} to the network!")
                st.session_state.show_add_person_modal = False
                st.rerun()
            else:
                st.error("Please fill in all required fields (*)")
    
    if st.button("❌ Cancel", key="cancel_add_person"):
        st.session_state.show_add_person_modal = False
        st.rerun()

# --- ADD FIRM MODAL ---
if st.session_state.show_add_firm_modal:
    st.markdown("---")
    st.subheader("🏢 Add New Hedge Fund")
    
    with st.form("add_firm_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            firm_name = st.text_input("Firm Name*", placeholder="Hillhouse Capital")
            location = st.selectbox("Primary Location*", 
                                  options=["", "Hong Kong", "Singapore", "Tokyo", "Seoul", "Mumbai", "Shanghai", "Beijing", "Taipei", "Bangkok", "Jakarta"])
            headquarters = st.text_input("Headquarters", placeholder="Beijing, China")
            aum = st.text_input("Assets Under Management", placeholder="60B USD")
            
        with col2:
            founded = st.number_input("Founded Year", min_value=1900, max_value=2025, value=2000)
            strategy = st.selectbox("Strategy", 
                                  options=["", "Long/Short Equity", "Multi-Strategy", "Quantitative", "Long-only", "Market Neutral", "Event Driven", "Macro"])
            website = st.text_input("Website", placeholder="https://company.com")
            
        description = st.text_area("Description", placeholder="Brief description of the hedge fund...")
        
        submitted = st.form_submit_button("Add Firm")
        
        if submitted:
            if firm_name and location:
                new_firm_id = str(uuid.uuid4())
                st.session_state.firms.append({
                    "id": new_firm_id,
                    "name": firm_name,
                    "location": location,
                    "headquarters": headquarters,
                    "aum": aum,
                    "founded": founded if founded > 1900 else None,
                    "strategy": strategy,
                    "website": website,
                    "description": description
                })
                
                st.success(f"✅ Added {firm_name} to the network!")
                st.session_state.show_add_firm_modal = False
                st.rerun()
            else:
                st.error("Please fill in Firm Name and Location")
    
    if st.button("❌ Cancel", key="cancel_add_firm"):
        st.session_state.show_add_firm_modal = False
        st.rerun()

# --- FIRMS VIEW ---
if st.session_state.current_view == 'firms':
    st.markdown("---")
    st.header("🏢 Hedge Funds in Asia")
    
    if not st.session_state.firms:
        st.info("No firms added yet. Use the 'Add Firm' button above.")
    else:
        # Firm cards
        cols = st.columns(2)
        for i, firm in enumerate(st.session_state.firms):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0;">
                        <h3>{firm['name']}</h3>
                        <p><strong>📍 Location:</strong> {firm['location']}</p>
                        <p><strong>💰 AUM:</strong> {firm['aum']}</p>
                        <p><strong>📈 Strategy:</strong> {firm['strategy']}</p>
                        <p><strong>👥 People:</strong> {len(get_people_by_firm(firm['name']))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"View {firm['name']}", key=f"view_firm_{firm['id']}"):
                        go_to_firm_details(firm['id'])
                        st.rerun()

# --- PEOPLE VIEW ---
elif st.session_state.current_view == 'people':
    st.markdown("---")
    st.header("👥 Professionals in Asian Hedge Funds")
    
    if not st.session_state.people:
        st.info("No people added yet. Use the 'Add Person' button above.")
    else:
        # Create DataFrame
        people_data = []
        for person in st.session_state.people:
            people_data.append({
                "Name": person['name'],
                "Title": person['current_title'],
                "Company": person['current_company_name'],
                "Location": person.get('location', 'Unknown'),
                "Strategy": person.get('strategy', 'Unknown'),
                "AUM Managed": person.get('aum_managed', ''),
                "Expertise": person.get('expertise', ''),
                "ID": person['id']
            })
        
        df = pd.DataFrame(people_data)
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            location_filter = st.selectbox("Filter by Location", ["All"] + list(df['Location'].unique()))
        with col2:
            company_filter = st.selectbox("Filter by Company", ["All"] + list(df['Company'].unique()))
        with col3:
            strategy_filter = st.selectbox("Filter by Strategy", ["All"] + list(df['Strategy'].unique()))
        with col4:
            expertise_filter = st.text_input("Search by Expertise", placeholder="Technology, Healthcare...")
        
        # Apply filters
        filtered_df = df.copy()
        if location_filter != "All":
            filtered_df = filtered_df[filtered_df['Location'] == location_filter]
        if company_filter != "All":
            filtered_df = filtered_df[filtered_df['Company'] == company_filter]
        if strategy_filter != "All":
            filtered_df = filtered_df[filtered_df['Strategy'] == strategy_filter]
        if expertise_filter:
            filtered_df = filtered_df[filtered_df['Expertise'].str.contains(expertise_filter, case=False, na=False)]
        
        st.dataframe(filtered_df.drop(columns=['ID']), use_container_width=True)
        
        # View buttons
        st.subheader("👤 View Individual Profiles")
        cols = st.columns(4)
        for i, person in enumerate(st.session_state.people):
            with cols[i % 4]:
                if st.button(f"View {person['name']}", key=f"view_person_{person['id']}"):
                    go_to_person_details(person['id'])
                    st.rerun()

# --- FIRM DETAILS VIEW ---
elif st.session_state.current_view == 'firm_details' and st.session_state.selected_firm_id:
    firm = get_firm_by_id(st.session_state.selected_firm_id)
    if not firm:
        st.error("Firm not found")
        go_to_firms()
        st.rerun()
    
    # Firm header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"🏢 {firm['name']}")
        st.markdown(f"**{firm['strategy']} Hedge Fund** • {firm['location']}")
    with col2:
        if st.button("← Back to Firms"):
            go_to_firms()
            st.rerun()
    
    # Firm details
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Assets Under Management", firm['aum'])
        st.metric("Founded", firm['founded'] if firm['founded'] else "Unknown")
        st.metric("Total Employees", len(get_people_by_firm(firm['name'])))
    
    with col2:
        st.markdown(f"**📍 Headquarters:** {firm['headquarters']}")
        st.markdown(f"**📈 Strategy:** {firm['strategy']}")
        if firm['website']:
            st.markdown(f"**🌐 Website:** [{firm['website']}]({firm['website']})")
    
    if firm['description']:
        st.markdown(f"**📄 Description:** {firm['description']}")
    
    # People at this firm
    st.markdown("---")
    st.subheader(f"👥 People at {firm['name']}")
    
    firm_people = get_people_by_firm(firm['name'])
    if firm_people:
        for person in firm_people:
            with st.expander(f"{person['name']} - {person['current_title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**📧 Email:** {person.get('email', 'Unknown')}")
                    st.write(f"**📱 Phone:** {person.get('phone', 'Unknown')}")
                    st.write(f"**🎓 Education:** {person.get('education', 'Unknown')}")
                with col2:
                    st.write(f"**🏆 Expertise:** {person.get('expertise', 'Unknown')}")
                    st.write(f"**💰 AUM Managed:** {person.get('aum_managed', 'Unknown')}")
                    if person.get('strategy') and person.get('strategy') != 'Unknown':
                        st.write(f"**📈 Strategy:** {person['strategy']}")
                    if person.get('linkedin_profile_url'):
                        st.markdown(f"**🔗 LinkedIn:** [{person['linkedin_profile_url']}]({person['linkedin_profile_url']})")
                
                if st.button(f"View Full Profile", key=f"view_full_{person['id']}"):
                    go_to_person_details(person['id'])
                    st.rerun()
    else:
        st.info("No people added for this firm yet.")

# --- PERSON DETAILS VIEW ---
elif st.session_state.current_view == 'person_details' and st.session_state.selected_person_id:
    person = get_person_by_id(st.session_state.selected_person_id)
    if not person:
        st.error("Person not found")
        go_to_people()
        st.rerun()
    
    # Person header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"👤 {person['name']}")
        st.markdown(f"**{person['current_title']}** at **{person['current_company_name']}**")
        st.markdown(f"📍 {person['location']}")
    with col2:
        if st.button("← Back to People"):
            go_to_people()
            st.rerun()
    
    # Contact info
    col1, col2 = st.columns(2)
    with col1:
        if person.get('email'):
            st.markdown(f"📧 [{person['email']}](mailto:{person['email']})")
        if person.get('phone'):
            st.markdown(f"📱 {person['phone']}")
        if person.get('linkedin_profile_url'):
            st.markdown(f"🔗 [LinkedIn Profile]({person['linkedin_profile_url']})")
    
    with col2:
        if person.get('education'):
            st.markdown(f"🎓 **Education:** {person['education']}")
        if person.get('expertise'):
            st.markdown(f"🏆 **Expertise:** {person['expertise']}")
        if person.get('aum_managed'):
            st.markdown(f"💰 **AUM Managed:** {person['aum_managed']}")
        if person.get('strategy') and person.get('strategy') != 'Unknown':
            st.markdown(f"📈 **Strategy:** {person['strategy']}")
    
    # Employment History
    st.markdown("---")
    st.subheader("💼 Employment History")
    
    employments = get_employments_by_person_id(person['id'])
    if employments:
        for emp in sorted(employments, key=lambda x: x['start_date'], reverse=True):
            end_date_str = emp['end_date'].strftime("%Y-%m-%d") if emp['end_date'] else "Present"
            duration = f"{emp['start_date'].strftime('%Y-%m-%d')} → {end_date_str}"
            
            st.markdown(f"""
            **{emp['title']}** at **{emp['company_name']}**  
            📅 {duration} • 📍 {emp.get('location', 'Unknown')}
            """)
    
    # Shared Work History
    st.markdown("---")
    st.subheader("🤝 Shared Work History")
    
    shared_history_data = []
    selected_person_employments = get_employments_by_person_id(person['id'])
    
    for other_person in st.session_state.people:
        if other_person['id'] == person['id']:
            continue
        
        other_person_employments = get_employments_by_person_id(other_person['id'])
        
        for selected_emp in selected_person_employments:
            for other_emp in other_person_employments:
                if selected_emp['company_name'] == other_emp['company_name']:
                    overlap = calculate_overlap_years(
                        selected_emp['start_date'], selected_emp['end_date'],
                        other_emp['start_date'], other_emp['end_date']
                    )
                    if overlap > 0:
                        shared_history_data.append({
                            "Name": other_person['name'],
                            "Shared Company": selected_emp['company_name'],
                            "Current Company": other_person['current_company_name'],
                            "Current Title": other_person['current_title'],
                            "Overlap Years": overlap
                        })
    
    if shared_history_data:
        df_shared = pd.DataFrame(shared_history_data).drop_duplicates().sort_values('Overlap Years', ascending=False)
        st.dataframe(df_shared, use_container_width=True)
    else:
        st.info("No shared work history found.")

# --- Footer ---
st.markdown("---")
st.markdown("### 🌏 Asian Hedge Fund Talent Intelligence Platform")
st.markdown("**Powered by:** AI Extraction • Professional Networks • Market Intelligence")
