import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time
from pathlib import Path

# Try to import google.generativeai, handle if not available
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Asian Hedge Fund Talent Map",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Persistence Setup ---
DATA_DIR = Path("hedge_fund_data")
DATA_DIR.mkdir(exist_ok=True)

PEOPLE_FILE = DATA_DIR / "people.json"
FIRMS_FILE = DATA_DIR / "firms.json"
EMPLOYMENTS_FILE = DATA_DIR / "employments.json"
EXTRACTIONS_FILE = DATA_DIR / "extractions.json"

def save_data():
    """Save all data to JSON files"""
    try:
        with open(PEOPLE_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.people, f, indent=2, default=str)
        
        with open(FIRMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.firms, f, indent=2, default=str)
        
        with open(EMPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.employments, f, indent=2, default=str)
        
        if 'all_extractions' in st.session_state:
            with open(EXTRACTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.all_extractions, f, indent=2, default=str)
        
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def load_data():
    """Load data from JSON files"""
    try:
        if PEOPLE_FILE.exists():
            with open(PEOPLE_FILE, 'r', encoding='utf-8') as f:
                people = json.load(f)
        else:
            people = []
        
        if FIRMS_FILE.exists():
            with open(FIRMS_FILE, 'r', encoding='utf-8') as f:
                firms = json.load(f)
        else:
            firms = []
        
        if EMPLOYMENTS_FILE.exists():
            with open(EMPLOYMENTS_FILE, 'r', encoding='utf-8') as f:
                employments = json.load(f)
                # Convert date strings back to date objects
                for emp in employments:
                    if emp.get('start_date'):
                        emp['start_date'] = datetime.strptime(emp['start_date'], '%Y-%m-%d').date()
                    if emp.get('end_date'):
                        emp['end_date'] = datetime.strptime(emp['end_date'], '%Y-%m-%d').date()
        else:
            employments = []
        
        if EXTRACTIONS_FILE.exists():
            with open(EXTRACTIONS_FILE, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
        else:
            extractions = []
        
        return people, firms, employments, extractions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], [], [], []

# --- Initialize Session State with Rich Dummy Data ---
def init_dummy_data():
    """Initialize with comprehensive dummy data if no saved data exists"""
    
    # Sample people with detailed backgrounds
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
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Raj Patel",
            "current_title": "Managing Director",
            "current_company_name": "Two Sigma Asia",
            "location": "Singapore",
            "email": "r.patel@twosigma.com",
            "linkedin_profile_url": "https://linkedin.com/in/rajpatel",
            "phone": "+65-8765-4321",
            "education": "IIT Delhi, Stanford MBA",
            "expertise": "Machine Learning, Algorithmic Trading",
            "aum_managed": "3.2B USD",
            "strategy": "Systematic Trading"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Michael Wong",
            "current_title": "Senior Vice President",
            "current_company_name": "Bridgewater Asia",
            "location": "Hong Kong",
            "email": "m.wong@bridgewater.com",
            "linkedin_profile_url": "https://linkedin.com/in/michaelwong",
            "phone": "+852-9876-1234",
            "education": "HKU, Chicago Booth",
            "expertise": "Global Macro, Risk Management",
            "aum_managed": "1.5B USD",
            "strategy": "Pure Alpha"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Jennifer Zhang",
            "current_title": "Portfolio Manager",
            "current_company_name": "Goldman Sachs Asia",
            "location": "Beijing",
            "email": "j.zhang@gs.com",
            "linkedin_profile_url": "https://linkedin.com/in/jenniferzhang",
            "phone": "+86-138-0013-8000",
            "education": "Peking University, Columbia Business School",
            "expertise": "Consumer, Internet",
            "aum_managed": "1.2B USD",
            "strategy": "Principal Investing"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "David Yamamoto",
            "current_title": "Head of Japan Equity",
            "current_company_name": "Citadel Asia",
            "location": "Tokyo",
            "email": "d.yamamoto@citadel.com",
            "linkedin_profile_url": "https://linkedin.com/in/davidyamamoto",
            "phone": "+81-3-1234-5678",
            "education": "Waseda University, Kellogg",
            "expertise": "Japanese Equities, Value Investing",
            "aum_managed": "950M USD",
            "strategy": "Equity Long/Short"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Priya Sharma",
            "current_title": "Director",
            "current_company_name": "Two Sigma Asia",
            "location": "Mumbai",
            "email": "p.sharma@twosigma.com",
            "linkedin_profile_url": "https://linkedin.com/in/priyasharma",
            "phone": "+91-98765-43210",
            "education": "IIM Ahmedabad, MIT",
            "expertise": "Data Science, Alternative Data",
            "aum_managed": "600M USD",
            "strategy": "Alternative Data"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "James Liu",
            "current_title": "Investment Director",
            "current_company_name": "Hillhouse Capital",
            "location": "Shanghai",
            "email": "j.liu@hillhouse.com",
            "linkedin_profile_url": "https://linkedin.com/in/jamesliu",
            "phone": "+86-21-1234-5678",
            "education": "CEIBS, Wharton",
            "expertise": "TMT, Private Equity",
            "aum_managed": "2.1B USD",
            "strategy": "Growth Equity"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Emily Tan",
            "current_title": "VP of Trading",
            "current_company_name": "Millennium Partners Asia",
            "location": "Hong Kong",
            "email": "e.tan@millennium.com",
            "linkedin_profile_url": "https://linkedin.com/in/emilytan",
            "phone": "+852-5555-6666",
            "education": "HKUST, London Business School",
            "expertise": "Systematic Trading, Risk Management",
            "aum_managed": "750M USD",
            "strategy": "Quantitative Equity"
        }
    ]
    
    # Sample firms with detailed information
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
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Two Sigma Asia",
            "location": "Singapore",
            "headquarters": "New York, USA",
            "aum": "25B USD",
            "founded": 2001,
            "strategy": "Systematic, Machine Learning",
            "website": "https://twosigma.com",
            "description": "Data-driven investment manager with Asian expansion"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Bridgewater Asia",
            "location": "Hong Kong",
            "headquarters": "Connecticut, USA",
            "aum": "140B USD",
            "founded": 1975,
            "strategy": "Global Macro, Pure Alpha",
            "website": "https://bridgewater.com",
            "description": "World's largest hedge fund with Asian investment focus"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Goldman Sachs Asia",
            "location": "Hong Kong",
            "headquarters": "New York, USA",
            "aum": "80B USD",
            "founded": 1869,
            "strategy": "Multi-strategy, Principal Investing",
            "website": "https://gs.com",
            "description": "Global investment bank with substantial Asian hedge fund operations"
        }
    ]
    
    # Create employment history with overlaps
    sample_employments = []
    
    # Li Wei Chen's history (Hillhouse Capital)
    li_id = sample_people[0]['id']
    sample_employments.extend([
        {
            "id": str(uuid.uuid4()),
            "person_id": li_id,
            "company_name": "Goldman Sachs Asia",
            "title": "Vice President",
            "start_date": date(2018, 3, 1),
            "end_date": date(2021, 8, 15),
            "location": "Hong Kong",
            "strategy": "Investment Banking"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": li_id,
            "company_name": "Hillhouse Capital",
            "title": "Portfolio Manager",
            "start_date": date(2021, 9, 1),
            "end_date": None,
            "location": "Hong Kong",
            "strategy": "Growth Equity"
        }
    ])
    
    # Jennifer Zhang's history (overlaps with Li Wei at Goldman Sachs)
    jennifer_id = sample_people[5]['id']
    sample_employments.extend([
        {
            "id": str(uuid.uuid4()),
            "person_id": jennifer_id,
            "company_name": "Goldman Sachs Asia",
            "title": "Associate",
            "start_date": date(2019, 6, 1),
            "end_date": date(2022, 12, 31),
            "location": "Hong Kong",
            "strategy": "Principal Investing"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": jennifer_id,
            "company_name": "Goldman Sachs Asia",
            "title": "Portfolio Manager",
            "start_date": date(2023, 1, 1),
            "end_date": None,
            "location": "Beijing",
            "strategy": "Principal Investing"
        }
    ])
    
    # Akira Tanaka's history
    akira_id = sample_people[1]['id']
    sample_employments.extend([
        {
            "id": str(uuid.uuid4()),
            "person_id": akira_id,
            "company_name": "Two Sigma Asia",
            "title": "Senior Researcher",
            "start_date": date(2017, 1, 15),
            "end_date": date(2020, 6, 30),
            "location": "Singapore",
            "strategy": "Systematic Trading"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": akira_id,
            "company_name": "Millennium Partners Asia",
            "title": "Chief Investment Officer",
            "start_date": date(2020, 7, 15),
            "end_date": None,
            "location": "Singapore",
            "strategy": "Multi-Strategy"
        }
    ])
    
    # Priya Sharma's history (overlaps with Akira at Two Sigma)
    priya_id = sample_people[7]['id']
    sample_employments.extend([
        {
            "id": str(uuid.uuid4()),
            "person_id": priya_id,
            "company_name": "Two Sigma Asia",
            "title": "Analyst",
            "start_date": date(2018, 8, 1),
            "end_date": date(2021, 3, 31),
            "location": "Singapore",
            "strategy": "Alternative Data"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": priya_id,
            "company_name": "Two Sigma Asia",
            "title": "Director",
            "start_date": date(2021, 4, 1),
            "end_date": None,
            "location": "Mumbai",
            "strategy": "Alternative Data"
        }
    ])
    
    # David Yamamoto and Sarah Kim (both at Citadel Asia with overlap)
    david_id = sample_people[6]['id']
    sarah_id = sample_people[2]['id']
    
    sample_employments.extend([
        {
            "id": str(uuid.uuid4()),
            "person_id": david_id,
            "company_name": "Bridgewater Asia",
            "title": "Associate",
            "start_date": date(2016, 4, 1),
            "end_date": date(2019, 9, 30),
            "location": "Hong Kong",
            "strategy": "Global Macro"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": david_id,
            "company_name": "Citadel Asia",
            "title": "Head of Japan Equity",
            "start_date": date(2019, 10, 15),
            "end_date": None,
            "location": "Tokyo",
            "strategy": "Equity Long/Short"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": sarah_id,
            "company_name": "Citadel Asia",
            "title": "Research Analyst",
            "start_date": date(2020, 2, 1),
            "end_date": date(2022, 6, 15),
            "location": "Seoul",
            "strategy": "Equity Research"
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": sarah_id,
            "company_name": "Citadel Asia",
            "title": "Head of Research",
            "start_date": date(2022, 6, 16),
            "end_date": None,
            "location": "Seoul",
            "strategy": "Equity Long/Short"
        }
    ])
    
    # Add more employment histories for remaining people
    for person in sample_people[3:]:  # Raj Patel onwards
        if person['id'] not in [p['person_id'] for p in sample_employments]:
            # Add current employment
            sample_employments.append({
                "id": str(uuid.uuid4()),
                "person_id": person['id'],
                "company_name": person['current_company_name'],
                "title": person['current_title'],
                "start_date": date(2020, 1, 1),
                "end_date": None,
                "location": person['location'],
                "strategy": person.get('strategy', 'Unknown')
            })
    
    return sample_people, sample_firms, sample_employments

def initialize_session_state():
    """Initialize session state with saved or dummy data"""
    people, firms, employments, extractions = load_data()
    
    # If no saved data, use dummy data
    if not people and not firms:
        people, firms, employments = init_dummy_data()
    
    if 'people' not in st.session_state:
        st.session_state.people = people
    if 'firms' not in st.session_state:
        st.session_state.firms = firms
    if 'employments' not in st.session_state:
        st.session_state.employments = employments
    if 'all_extractions' not in st.session_state:
        st.session_state.all_extractions = extractions
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'firms'
    if 'selected_person_id' not in st.session_state:
        st.session_state.selected_person_id = None
    if 'selected_firm_id' not in st.session_state:
        st.session_state.selected_firm_id = None
    if 'show_add_person_modal' not in st.session_state:
        st.session_state.show_add_person_modal = False
    if 'show_add_firm_modal' not in st.session_state:
        st.session_state.show_add_firm_modal = False

# --- AI Setup ---
@st.cache_resource
def setup_gemini(api_key, model_id="gemini-1.5-flash"):
    """Setup Gemini AI model safely with model selection"""
    if not GENAI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        # Store model_id as an attribute for rate limiting
        model.model_id = model_id
        return model
    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None

def extract_talent_simple(text, model):
    """Crash-proof extraction with minimal UI updates"""
    if not model:
        return []
    
    # Simple size-based processing decision
    max_single_chunk = 15000
    
    if len(text) <= max_single_chunk:
        # Single chunk - simple and reliable
        st.info("📄 Processing as single chunk...")
        return extract_single_chunk_safe(text, model)
    else:
        # Multi-chunk with crash protection
        st.info(f"📊 Large file detected ({len(text):,} chars). Using safe chunking...")
        return extract_multi_chunk_safe(text, model, max_single_chunk)

def extract_single_chunk_safe(text, model):
    """Safe single chunk extraction with timeout"""
    try:
        prompt = f"""Extract hedge fund talent movements from this text. Return JSON only:

{text}

{{"people": [{{"name": "Full Name", "company": "Company", "title": "Position", "movement_type": "hire|promotion|launch|departure", "location": "Location"}}]}}

Find ALL people in professional contexts."""
        
        response = model.generate_content(prompt)
        if not response or not response.text:
            return []
        
        # Parse JSON
        json_start = response.text.find('{')
        json_end = response.text.rfind('}') + 1
        
        if json_start == -1:
            return []
        
        result = json.loads(response.text[json_start:json_end])
        people = result.get('people', [])
        
        # Filter valid entries
        return [p for p in people if p.get('name') and p.get('company')]
        
    except Exception as e:
        st.warning(f"Single chunk failed: {str(e)[:100]}")
        return []

def extract_multi_chunk_safe(text, model, chunk_size):
    """Crash-proof multi-chunk processing with minimal UI"""
    
    try:
        # Simple chunking without complex overlap logic
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # Skip tiny chunks
                chunks.append(chunk)
        
        st.info(f"Split into {len(chunks)} chunks. Processing with delays...")
        
        # Conservative delay based on model
        model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
        if '1.5-pro' in model_id:
            delay = 40  # Very conservative
        else:
            delay = 10  # Conservative for all Flash models
        
        # Process chunks with minimal UI updates
        all_extractions = []
        successful = 0
        failed = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Simple status without fancy progress bars
                if i > 0:
                    st.info(f"Rate limiting delay: {delay}s...")
                    time.sleep(delay)
                
                st.info(f"Processing chunk {i+1}/{len(chunks)}...")
                
                # Extract from chunk
                chunk_results = extract_single_chunk_safe(chunk, model)
                
                if chunk_results:
                    # Simple deduplication
                    for result in chunk_results:
                        name_company = f"{result.get('name', '').lower()}|{result.get('company', '').lower()}"
                        
                        # Check if already exists
                        exists = any(
                            f"{existing.get('name', '').lower()}|{existing.get('company', '').lower()}" == name_company 
                            for existing in all_extractions
                        )
                        
                        if not exists:
                            all_extractions.append(result)
                    
                    successful += 1
                    st.success(f"✅ Chunk {i+1}: Found {len(chunk_results)} people")
                else:
                    failed += 1
                    st.warning(f"⚠️ Chunk {i+1}: No results")
                
                # Safety: Stop if too many failures
                if failed > 3 and failed > successful:
                    st.error("Too many chunk failures. Stopping to prevent issues.")
                    break
                    
            except Exception as chunk_error:
                failed += 1
                st.error(f"❌ Chunk {i+1} failed: {str(chunk_error)[:100]}")
                
                # Stop on API errors
                if "rate" in str(chunk_error).lower() or "quota" in str(chunk_error).lower():
                    st.error("Rate limit hit. Stopping processing.")
                    break
                
                # Continue for other errors
                continue
        
        # Final summary
        st.info(f"Completed: {successful} successful, {failed} failed chunks")
        st.success(f"Total unique extractions: {len(all_extractions)}")
        
        return all_extractions
        
    except Exception as e:
        st.error(f"Multi-chunk processing failed: {e}")
        return []

def extract_single_chunk(text, model):
    """Extract from a single chunk"""
    prompt = f"""
Extract hedge fund talent movements from this newsletter text.

TEXT:
{text}

Return JSON format:
{{
  "people": [
    {{
      "name": "Full Name",
      "company": "Company Name", 
      "title": "Job Title",
      "movement_type": "hire|promotion|launch|departure",
      "location": "Location",
      "previous_company": "Previous Company (if mentioned)"
    }}
  ]
}}

Find ALL people mentioned in professional contexts. Look for hires, promotions, fund launches, departures.
"""
    
    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            return []
        
        # Parse JSON response
        json_start = response.text.find('{')
        json_end = response.text.rfind('}') + 1
        
        if json_start == -1:
            return []
        
        json_text = response.text[json_start:json_end]
        result = json.loads(json_text)
        
        people = result.get('people', [])
        # Filter valid entries
        valid_people = []
        for person in people:
            if person.get('name') and person.get('company'):
                valid_people.append(person)
        
        return valid_people
        
    except Exception as e:
        st.error(f"Extraction error: {e}")
        return []

def extract_with_smart_chunking(text, model, chunk_size):
    """Smart chunking with robust error handling and conservative rate limiting"""
    
    # Split text into chunks with overlap
    chunks = []
    overlap = 1000  # 1K character overlap to catch names at boundaries
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > len(chunk) * 0.8:  # Only if near end
                chunk = chunk[:last_period + 1]
                end = start + len(chunk)
        
        chunks.append(chunk)
        start = end - overlap
        
        if start >= len(text):
            break
    
    st.info(f"📊 Processing {len(chunks)} chunks (~{chunk_size/1000:.0f}K chars each)")
    
    # Conservative rate limiting based on selected model
    model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
    
    if '1.5-pro' in model_id:
        delay = 45  # Extra conservative for Pro (was 35)
        rpm = "2 RPM"
        timeout = 60  # Longer timeout for Pro
    elif '2.5-flash' in model_id:
        delay = 10  # More conservative (was 7)
        rpm = "10 RPM"
        timeout = 45
    elif '2.0-flash' in model_id:
        delay = 8   # More conservative (was 5)
        rpm = "15 RPM"
        timeout = 30
    else:  # 1.5-flash and others
        delay = 8   # More conservative (was 5)
        rpm = "15 RPM"
        timeout = 30
    
    st.info(f"⚡ Using {model_id} | Rate limit: {rpm} ({delay}s delay) | Timeout: {timeout}s per chunk")
    
    # Estimate total time and warn user
    total_estimated_time = len(chunks) * (delay + 10)  # delay + processing time
    if total_estimated_time > 300:  # > 5 minutes
        st.warning(f"⏰ **Estimated processing time: {total_estimated_time//60}min {total_estimated_time%60}s**")
        st.warning("💡 Consider using a faster model (1.5 Flash) or breaking up the newsletter")
        
        if not st.checkbox("✅ I understand this will take time and want to proceed"):
            st.stop()
    
    # Process chunks with robust error handling
    all_extractions = []
    seen_names = set()  # Deduplicate across chunks
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_container = st.empty()
    
    successful_chunks = 0
    failed_chunks = 0
    
    for i, chunk in enumerate(chunks):
        try:
            # Update progress
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            status_text.info(f"🔄 Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)...")
            
            # Rate limiting delay (except first chunk)
            if i > 0:
                status_text.info(f"⏱️ Rate limiting delay: {delay}s...")
                time.sleep(delay)
            
            # Process chunk with timeout and retry logic
            max_retries = 2
            chunk_extractions = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Add timeout handling
                    start_time = time.time()
                    chunk_extractions = extract_single_chunk(chunk, model)
                    elapsed = time.time() - start_time
                    
                    if elapsed > timeout:
                        raise Exception(f"Timeout: took {elapsed:.1f}s (max: {timeout}s)")
                    
                    # Success!
                    break
                    
                except Exception as chunk_error:
                    if attempt < max_retries:
                        error_container.warning(f"⚠️ Chunk {i+1} attempt {attempt+1} failed: {str(chunk_error)[:100]}... Retrying in {delay}s...")
                        time.sleep(delay)  # Extra delay before retry
                    else:
                        # Final failure
                        error_container.error(f"❌ Chunk {i+1} failed after {max_retries + 1} attempts")
                        failed_chunks += 1
                        chunk_extractions = []
                        break
            
            if chunk_extractions is None:
                chunk_extractions = []
            
            # Deduplicate and add successful extractions
            if chunk_extractions:
                new_count = 0
                for extraction in chunk_extractions:
                    name = extraction.get('name', '').strip().lower()
                    company = extraction.get('company', '').strip().lower()
                    key = f"{name}|{company}"
                    
                    if key not in seen_names and name and company:
                        seen_names.add(key)
                        all_extractions.append(extraction)
                        new_count += 1
                
                successful_chunks += 1
                status_text.success(f"✅ Chunk {i+1}: Found {len(chunk_extractions)} entries ({new_count} new, {len(all_extractions)} total)")
            else:
                if failed_chunks == 0:  # Only count as failed if no retries worked
                    failed_chunks += 1
                status_text.warning(f"⚠️ Chunk {i+1}: No extractions found")
            
            # Early exit if too many failures
            if failed_chunks > len(chunks) // 2:  # More than 50% failed
                error_container.error(f"🚫 **Too many chunk failures ({failed_chunks}/{i+1})**. Stopping to prevent further issues.")
                error_container.info("💡 **Suggestions**: Try a different model, reduce file size, or check your API key")
                break
                
        except KeyboardInterrupt:
            status_text.warning("⏹️ **Processing stopped by user**")
            break
            
        except Exception as e:
            failed_chunks += 1
            error_msg = str(e)
            
            # Handle specific error types
            if "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                error_container.error(f"🚫 **Rate Limit Hit on Chunk {i+1}**: {error_msg}")
                error_container.info("💡 **Solution**: Wait a few minutes and try again, or use a different model")
                break
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                error_container.error(f"🔑 **API Key Error**: {error_msg}")
                break
            else:
                error_container.error(f"❌ **Unexpected Error on Chunk {i+1}**: {error_msg}")
                
                # Continue with next chunk for unexpected errors
                continue
    
    # Final progress update
    progress_bar.progress(1.0)
    
    # Results summary
    if successful_chunks > 0:
        success_rate = (successful_chunks / len(chunks)) * 100
        status_text.success(f"""
        🎯 **Processing Complete!**
        ✅ Successful chunks: {successful_chunks}/{len(chunks)} ({success_rate:.1f}%)
        ❌ Failed chunks: {failed_chunks}/{len(chunks)}
        📊 Total unique extractions: {len(all_extractions)}
        """)
        
        if failed_chunks > 0:
            error_container.warning(f"⚠️ **Partial Success**: {failed_chunks} chunks failed. Results may be incomplete.")
    else:
        status_text.error("❌ **All chunks failed!** Check rate limits, API key, or try a different model.")
        error_container.info("🔧 **Troubleshooting**: Try 1.5 Flash model, check API key, or reduce file size")
    
    # Clean up UI after a delay
    time.sleep(3)
    if successful_chunks > 0:  # Only clean up if we got some results
        progress_bar.empty()
        status_text.empty()
        error_container.empty()
    
    return all_extractions

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
    """Calculate overlap between two employment periods"""
    today = date.today()
    period1_end = end1 if end1 is not None else today
    period2_end = end2 if end2 is not None else today
    
    latest_start = max(start1, start2)
    earliest_end = min(period1_end, period2_end)
    
    overlap_days = (earliest_end - latest_start).days
    if overlap_days <= 0:
        return 0.0
    return round(overlap_days / 365.25, 1)

def get_shared_work_history(person_id):
    """Get people who worked at same companies with overlap periods"""
    person_employments = get_employments_by_person_id(person_id)
    shared_history = []
    
    for other_person in st.session_state.people:
        if other_person['id'] == person_id:
            continue
        
        other_employments = get_employments_by_person_id(other_person['id'])
        
        for person_emp in person_employments:
            for other_emp in other_employments:
                if person_emp['company_name'] == other_emp['company_name']:
                    overlap = calculate_overlap_years(
                        person_emp['start_date'], person_emp['end_date'],
                        other_emp['start_date'], other_emp['end_date']
                    )
                    if overlap > 0:
                        shared_history.append({
                            "colleague_name": other_person['name'],
                            "colleague_id": other_person['id'],
                            "shared_company": person_emp['company_name'],
                            "colleague_current_company": other_person['current_company_name'],
                            "colleague_current_title": other_person['current_title'],
                            "overlap_years": overlap,
                            "person_title": person_emp['title'],
                            "colleague_title": other_emp['title']
                        })
    
    # Remove duplicates and sort by overlap
    unique_shared = {}
    for item in shared_history:
        key = f"{item['colleague_id']}_{item['shared_company']}"
        if key not in unique_shared or item['overlap_years'] > unique_shared[key]['overlap_years']:
            unique_shared[key] = item
    
    return sorted(unique_shared.values(), key=lambda x: x['overlap_years'], reverse=True)

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

# Initialize session state
initialize_session_state()

# --- SIDEBAR: AI Talent Extractor ---
with st.sidebar:
    st.title("🤖 AI Talent Extractor")
    
    # API Key Setup
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            st.success("✅ API key loaded from secrets")
    except:
        pass
    
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password", 
                              help="Get from: https://makersuite.google.com/app/apikey")
    
    # Model Selection
    st.markdown("---")
    st.subheader("🤖 Model Selection")
    
    model_options = {
        "Gemini 1.5 Flash (Recommended)": "gemini-1.5-flash",
        "Gemini 1.5 Pro (Advanced)": "gemini-1.5-pro", 
        "Gemini 2.0 Flash": "gemini-2.0-flash-exp",
        "Gemini 2.5 Flash (Latest)": "gemini-2.5-flash-exp"
    }
    
    selected_model_name = st.selectbox(
        "Choose AI model:",
        options=list(model_options.keys()),
        index=0,  # Default to 1.5 Flash
        help="Different models have different capabilities and rate limits"
    )
    
    selected_model_id = model_options[selected_model_name]
    
    # Show model info
    if "1.5-flash" in selected_model_id:
        st.success("⚡ **Fast & Reliable**: 15 requests/min, good extraction quality")
        rate_info = "15 RPM (5s delay)"
    elif "1.5-pro" in selected_model_id:
        st.warning("🧠 **Most Advanced**: 2 requests/min, best for complex newsletters")
        rate_info = "2 RPM (35s delay) - SLOW but thorough"
    elif "2.0-flash" in selected_model_id:
        st.info("🔥 **Balanced**: 15 requests/min, improved accuracy over 1.5")
        rate_info = "15 RPM (5s delay)"
    elif "2.5-flash" in selected_model_id:
        st.info("🌟 **Latest**: 10 requests/min, cutting-edge capabilities")
        rate_info = "10 RPM (7s delay)"
    
    st.caption(f"📊 Rate limit: {rate_info}")
    
    # Setup model with selected version
    model = None
    if api_key and GENAI_AVAILABLE:
        model = setup_gemini(api_key, selected_model_id)
        
        # Show rate limit info
        if model:
            model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
            
            with st.expander("⚡ Rate Limits & Processing Info"):
                if '1.5-pro' in model_id:
                    st.warning("🧠 **Gemini 1.5 Pro**: 2 requests/min (35s between chunks)")
                    st.write("• Best for complex analysis")
                    st.write("• Slowest processing due to low rate limit")
                    st.write("• Most thorough extraction")
                elif '2.5-flash' in model_id:
                    st.info("🌟 **Gemini 2.5 Flash**: 10 requests/min (7s between chunks)")
                    st.write("• Latest model with cutting-edge capabilities")
                    st.write("• Good balance of speed and quality")
                elif '2.0-flash' in model_id:
                    st.success("🔥 **Gemini 2.0 Flash**: 15 requests/min (5s between chunks)")
                    st.write("• Improved accuracy over 1.5")
                    st.write("• Fast processing")
                else:  # 1.5-flash
                    st.success("⚡ **Gemini 1.5 Flash**: 15 requests/min (5s between chunks)")
                    st.write("• Fastest processing")
                    st.write("• Reliable and stable")
                    st.write("• Recommended for most newsletters")
                
                st.markdown("**📊 Processing Time Examples:**")
                if '1.5-pro' in model_id:
                    st.write("• Small newsletter (0-15K chars): ~45 seconds")
                    st.write("• Medium newsletter (15-45K chars): 2-8 minutes") 
                    st.write("• Large newsletter (45K+ chars): 8-20 minutes")
                elif '2.5-flash' in model_id:
                    st.write("• Small newsletter (0-15K chars): ~35 seconds")
                    st.write("• Medium newsletter (15-45K chars): 1-4 minutes") 
                    st.write("• Large newsletter (45K+ chars): 4-10 minutes")
                else:  # Flash models
                    st.write("• Small newsletter (0-15K chars): ~30 seconds")
                    st.write("• Medium newsletter (15-45K chars): 1-3 minutes") 
                    st.write("• Large newsletter (45K+ chars): 3-8 minutes")
                
                st.info("💡 **Tip**: Use 1.5 Flash for speed, 1.5 Pro for accuracy, 2.5 Flash for latest features")
        
        st.markdown("---")
        st.subheader("📄 Extract from Newsletter")
        
        # Input method - simplified file handling
        input_method = st.radio("Input method:", ["📝 Text", "📁 File"])
        
        newsletter_text = ""
        if input_method == "📝 Text":
            newsletter_text = st.text_area("Newsletter content:", height=200, 
                                         placeholder="Paste hedge fund newsletter content here...")
        else:
            uploaded_file = st.file_uploader("Upload newsletter:", type=['txt'], 
                                            help="Recommended: under 50KB for reliable processing")
            if uploaded_file:
                try:
                    # Simple file size check
                    file_size = len(uploaded_file.getvalue())
                    file_size_kb = file_size / 1024
                    
                    if file_size_kb > 100:  # 100KB limit
                        st.error(f"File too large: {file_size_kb:.1f}KB. Please use a smaller file.")
                    else:
                        # Simple encoding
                        raw_data = uploaded_file.getvalue()
                        try:
                            newsletter_text = raw_data.decode('utf-8')
                        except:
                            try:
                                newsletter_text = raw_data.decode('latin-1')
                            except:
                                st.error("Could not read file. Try saving as UTF-8 text file.")
                        
                        if newsletter_text:
                            st.success(f"✅ File loaded: {file_size_kb:.1f}KB")
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Show processing info
        if newsletter_text:
            char_count = len(newsletter_text)
            if char_count > 15000:
                chunks_needed = max(1, char_count // 15000)
                st.warning(f"⚡ Large newsletter detected! Will use {chunks_needed} chunks with rate limiting")
                
                # Estimate processing time based on selected model
                if model:
                    model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
                    
                    if '1.5-pro' in model_id:
                        est_time = chunks_needed * 40  # 35s delay + processing
                        model_desc = "Gemini 1.5 Pro (slow but thorough)"
                    elif '2.5-flash' in model_id:
                        est_time = chunks_needed * 10  # 7s delay + processing
                        model_desc = "Gemini 2.5 Flash (balanced)"
                    else:  # Flash models
                        est_time = chunks_needed * 8   # 5s delay + processing
                        model_desc = "Gemini Flash (fast)"
                    
                    st.info(f"🕐 **{model_desc}**: ~{est_time//60}min {est_time%60}s estimated")
                    
                    if est_time > 300:  # > 5 minutes
                        st.warning("⚠️ Long processing time! Consider using a faster model for large newsletters.")
            else:
                st.success(f"✅ Single chunk processing ({char_count:,} chars)")
                if model:
                    model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
                    st.info(f"Using {model_id.replace('gemini-', '').replace('-exp', '').title()}")

        # Extract button - simplified and crash-proof
        if st.button("🚀 Extract Talent", use_container_width=True):
            if not newsletter_text.strip():
                st.error("Please provide newsletter content")
            elif not model:
                st.error("Please provide API key")
            else:
                char_count = len(newsletter_text)
                
                # Simple size checks
                if char_count > 150000:  # 150KB hard limit
                    st.error(f"File too large: {char_count:,} characters. Try a smaller file or copy/paste sections.")
                elif char_count > 50000:  # Warning for large files
                    st.warning(f"Large file: {char_count:,} characters. This may take 5-10 minutes.")
                    if not st.checkbox("I want to proceed anyway"):
                        st.stop()
                
                # Simple processing without complex error handling
                try:
                    st.info("🤖 Starting extraction...")
                    extractions = extract_talent_simple(newsletter_text, model)
                    
                    if extractions:
                        # Add metadata
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for ext in extractions:
                            ext['timestamp'] = timestamp
                            ext['id'] = str(uuid.uuid4())
                        
                        # Save results
                        st.session_state.all_extractions.extend(extractions)
                        save_data()
                        
                        # Show results
                        st.success(f"🎉 Found {len(extractions)} people!")
                        
                        # Simple preview
                        st.write("**Preview:**")
                        for i, ext in enumerate(extractions[:5]):
                            st.write(f"{i+1}. {ext['name']} → {ext['company']}")
                        
                        if len(extractions) > 5:
                            st.write(f"... and {len(extractions) - 5} more")
                    else:
                        st.warning("No people found. Try a different text or model.")
                        
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    st.info("Try: different model, smaller file, or copy/paste instead of upload")
        
        # Quick test
        if st.button("🧪 Test with Sample", use_container_width=True):
            sample = """
            Goldman Sachs veteran John Smith joins Citadel Asia as Managing Director in Hong Kong.
            Former JPMorgan portfolio manager Lisa Chen launches her own hedge fund, Dragon Capital, 
            focusing on Asian equities. Millennium Partners promotes Alex Wang to head of quant trading 
            in Singapore. Sarah Kim moves from Bridgewater to become CIO at newly formed Tiger Asia.
            """
            
            with st.spinner(f"Testing {selected_model_name}..."):
                extractions = extract_talent_simple(sample, model)
                st.write(f"**Test result:** {len(extractions)} people found")
                for ext in extractions:
                    st.write(f"• {ext['name']} → {ext['company']}")
        
        # Model comparison test
        if st.button("🔬 Compare All Models", use_container_width=True):
            if api_key:
                sample = """
                Goldman Sachs veteran John Smith joins Citadel Asia as Managing Director in Hong Kong.
                Former JPMorgan portfolio manager Lisa Chen launches her own hedge fund, Dragon Capital, 
                focusing on Asian equities. Millennium Partners promotes Alex Wang to head of quant trading 
                in Singapore. Sarah Kim moves from Bridgewater to become CIO at newly formed Tiger Asia.
                """
                
                st.write("**Testing all models on same sample:**")
                
                models_to_test = [
                    ("Gemini 1.5 Flash", "gemini-1.5-flash"),
                    ("Gemini 1.5 Pro", "gemini-1.5-pro"),
                    ("Gemini 2.0 Flash", "gemini-2.0-flash-exp"),
                    ("Gemini 2.5 Flash", "gemini-2.5-flash-exp")
                ]
                
                for model_name, model_id in models_to_test:
                    try:
                        test_model = setup_gemini(api_key, model_id)
                        if test_model:
                            with st.spinner(f"Testing {model_name}..."):
                                start_time = time.time()
                                extractions = extract_single_chunk(sample, test_model)
                                elapsed = time.time() - start_time
                                count = len(extractions) if extractions else 0
                                st.write(f"**{model_name}**: {count} people ({elapsed:.1f}s)")
                        else:
                            st.write(f"**{model_name}**: Setup failed")
                    except Exception as e:
                        st.write(f"**{model_name}**: Error - {str(e)}")
                    
                    # Small delay to avoid rate limits
                    time.sleep(2)
                
                st.info("💡 Higher extraction count usually = better model for your use case")
            else:
                st.error("Please provide API key first")
    
    elif not GENAI_AVAILABLE:
        st.error("Please install: pip install google-generativeai")
    
    # Show recent extractions
    if st.session_state.all_extractions:
        st.markdown("---")
        st.subheader("📊 Recent Extractions")
        st.metric("Total Extracted", len(st.session_state.all_extractions))
        
        # Add people from extractions
        if st.button("📥 Import All to Database", use_container_width=True):
            added_count = 0
            for ext in st.session_state.all_extractions:
                # Check if person already exists
                existing = any(p['name'].lower() == ext['name'].lower() 
                             for p in st.session_state.people)
                
                if not existing:
                    # Add person
                    new_person_id = str(uuid.uuid4())
                    st.session_state.people.append({
                        "id": new_person_id,
                        "name": ext['name'],
                        "current_title": ext.get('title', 'Unknown'),
                        "current_company_name": ext['company'],
                        "location": ext.get('location', 'Unknown'),
                        "email": "",
                        "linkedin_profile_url": "",
                        "phone": "",
                        "education": "",
                        "expertise": "",
                        "aum_managed": "",
                        "strategy": "Unknown"
                    })
                    
                    # Add firm if doesn't exist
                    if not get_firm_by_name(ext['company']):
                        st.session_state.firms.append({
                            "id": str(uuid.uuid4()),
                            "name": ext['company'],
                            "location": ext.get('location', 'Unknown'),
                            "headquarters": "Unknown",
                            "aum": "Unknown",
                            "founded": None,
                            "strategy": "Hedge Fund",
                            "website": "",
                            "description": f"Hedge fund - extracted from newsletter"
                        })
                    
                    # Add employment
                    st.session_state.employments.append({
                        "id": str(uuid.uuid4()),
                        "person_id": new_person_id,
                        "company_name": ext['company'],
                        "title": ext.get('title', 'Unknown'),
                        "start_date": date.today(),
                        "end_date": None,
                        "location": ext.get('location', 'Unknown'),
                        "strategy": "Unknown"
                    })
                    
                    added_count += 1
            
            save_data()  # Save changes
            st.success(f"✅ Added {added_count} new people to database!")
            st.rerun()

# --- MAIN CONTENT AREA ---
st.title("🏢 Asian Hedge Fund Talent Map")
st.markdown("### Professional network mapping for Asia's hedge fund industry")

# Auto-save indicator
if st.sidebar.button("💾 Save Data"):
    if save_data():
        st.sidebar.success("✅ Data saved!")
    else:
        st.sidebar.error("❌ Save failed!")

# Top Navigation
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

with col1:
    if st.button("🏢 Firms", use_container_width=True, 
                 type="primary" if st.session_state.current_view == 'firms' else "secondary"):
        go_to_firms()
        st.rerun()

with col2:
    if st.button("👥 People", use_container_width=True, 
                 type="primary" if st.session_state.current_view == 'people' else "secondary"):
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
    col5a, col5b = st.columns(2)
    with col5a:
        st.metric("People", len(st.session_state.people))
    with col5b:
        st.metric("Firms", len(st.session_state.firms))

# --- ADD PERSON MODAL ---
if st.session_state.show_add_person_modal:
    st.markdown("---")
    st.subheader("➕ Add New Person")
    
    with st.form("add_person_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", placeholder="John Smith")
            title = st.text_input("Current Title*", placeholder="Portfolio Manager")
            company = st.selectbox("Current Company*", 
                                 options=[""] + [f['name'] for f in st.session_state.firms])
            location = st.selectbox("Location*", 
                                  options=["", "Hong Kong", "Singapore", "Tokyo", "Seoul", 
                                          "Mumbai", "Shanghai", "Beijing", "Taipei"])
        
        with col2:
            email = st.text_input("Email", placeholder="john.smith@company.com")
            phone = st.text_input("Phone", placeholder="+852-1234-5678")
            education = st.text_input("Education", placeholder="Harvard, MIT")
            expertise = st.text_input("Expertise", placeholder="Equities, Technology")
        
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
                    "linkedin_profile_url": "",
                    "phone": phone,
                    "education": education,
                    "expertise": expertise,
                    "aum_managed": "",
                    "strategy": "Unknown"
                })
                
                # Add employment record
                st.session_state.employments.append({
                    "id": str(uuid.uuid4()),
                    "person_id": new_person_id,
                    "company_name": company,
                    "title": title,
                    "start_date": date.today(),
                    "end_date": None,
                    "location": location,
                    "strategy": "Unknown"
                })
                
                save_data()  # Auto-save
                st.success(f"✅ Added {name}!")
                st.session_state.show_add_person_modal = False
                st.rerun()
            else:
                st.error("Please fill required fields (*)")
    
    if st.button("❌ Cancel", key="cancel_add_person"):
        st.session_state.show_add_person_modal = False
        st.rerun()

# --- ADD FIRM MODAL ---
if st.session_state.show_add_firm_modal:
    st.markdown("---")
    st.subheader("🏢 Add New Firm")
    
    with st.form("add_firm_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            firm_name = st.text_input("Firm Name*", placeholder="Tiger Asia Management")
            location = st.selectbox("Location*", 
                                  options=["", "Hong Kong", "Singapore", "Tokyo", "Seoul", 
                                          "Mumbai", "Shanghai", "Beijing", "Taipei"])
            aum = st.text_input("AUM", placeholder="5B USD")
            
        with col2:
            strategy = st.selectbox("Strategy", 
                                  options=["", "Long/Short Equity", "Multi-Strategy", 
                                          "Quantitative", "Macro", "Event Driven"])
            founded = st.number_input("Founded", min_value=1900, max_value=2025, value=2000)
            website = st.text_input("Website", placeholder="https://company.com")
        
        submitted = st.form_submit_button("Add Firm")
        
        if submitted:
            if firm_name and location:
                st.session_state.firms.append({
                    "id": str(uuid.uuid4()),
                    "name": firm_name,
                    "location": location,
                    "headquarters": location,
                    "aum": aum,
                    "founded": founded if founded > 1900 else None,
                    "strategy": strategy,
                    "website": website,
                    "description": f"{strategy} hedge fund based in {location}"
                })
                
                save_data()  # Auto-save
                st.success(f"✅ Added {firm_name}!")
                st.session_state.show_add_firm_modal = False
                st.rerun()
            else:
                st.error("Please fill Firm Name and Location")
    
    if st.button("❌ Cancel", key="cancel_add_firm"):
        st.session_state.show_add_firm_modal = False
        st.rerun()

# --- FIRMS VIEW ---
if st.session_state.current_view == 'firms':
    st.markdown("---")
    st.header("🏢 Hedge Funds in Asia")
    
    if not st.session_state.firms:
        st.info("No firms added yet. Use 'Add Firm' button above.")
    else:
        # Create a grid layout for firm cards
        cols = st.columns(2)
        for i, firm in enumerate(st.session_state.firms):
            with cols[i % 2]:
                with st.container():
                    people_count = len(get_people_by_firm(firm['name']))
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f9f9f9;">
                        <h3 style="margin-top: 0;">{firm['name']}</h3>
                        <p><strong>📍 Location:</strong> {firm['location']}</p>
                        <p><strong>💰 AUM:</strong> {firm['aum']}</p>
                        <p><strong>📈 Strategy:</strong> {firm['strategy']}</p>
                        <p><strong>👥 People:</strong> {people_count}</p>
                        <p><strong>🏛️ Founded:</strong> {firm.get('founded', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"View Details", key=f"view_firm_{firm['id']}", use_container_width=True):
                        go_to_firm_details(firm['id'])
                        st.rerun()

# --- PEOPLE VIEW ---
elif st.session_state.current_view == 'people':
    st.markdown("---")
    st.header("👥 Hedge Fund Professionals")
    
    if not st.session_state.people:
        st.info("No people added yet. Use 'Add Person' button above.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            locations = ["All"] + sorted(list(set(p['location'] for p in st.session_state.people)))
            location_filter = st.selectbox("Filter by Location", locations)
        with col2:
            companies = ["All"] + sorted(list(set(p['current_company_name'] for p in st.session_state.people)))
            company_filter = st.selectbox("Filter by Company", companies)
        with col3:
            search_term = st.text_input("Search by Name", placeholder="Enter name...")
        
        # Apply filters
        filtered_people = st.session_state.people
        if location_filter != "All":
            filtered_people = [p for p in filtered_people if p['location'] == location_filter]
        if company_filter != "All":
            filtered_people = [p for p in filtered_people if p['current_company_name'] == company_filter]
        if search_term:
            filtered_people = [p for p in filtered_people if search_term.lower() in p['name'].lower()]
        
        # Display people in a grid
        st.write(f"**Showing {len(filtered_people)} people**")
        
        cols = st.columns(3)
        for i, person in enumerate(filtered_people):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 5px 0; background-color: #f8f9fa;">
                        <h4 style="margin-top: 0; color: #333;">{person['name']}</h4>
                        <p><strong>{person['current_title']}</strong></p>
                        <p>🏢 {person['current_company_name']}</p>
                        <p>📍 {person['location']}</p>
                        <p>💰 {person.get('aum_managed', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("View Profile", key=f"view_person_{person['id']}", use_container_width=True):
                        go_to_person_details(person['id'])
                        st.rerun()

# --- FIRM DETAILS VIEW ---
elif st.session_state.current_view == 'firm_details' and st.session_state.selected_firm_id:
    firm = get_firm_by_id(st.session_state.selected_firm_id)
    if not firm:
        st.error("Firm not found")
        go_to_firms()
        st.rerun()
    
    # Back button
    if st.button("← Back to Firms"):
        go_to_firms()
        st.rerun()
    
    st.header(f"🏢 {firm['name']}")
    
    # Firm details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Assets Under Management", firm['aum'])
    with col2:
        st.metric("Founded", firm.get('founded', 'Unknown'))
    with col3:
        people_count = len(get_people_by_firm(firm['name']))
        st.metric("Total People", people_count)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**📍 Location:** {firm['location']}")
        st.markdown(f"**🏛️ Headquarters:** {firm.get('headquarters', 'Unknown')}")
    with col2:
        st.markdown(f"**📈 Strategy:** {firm['strategy']}")
        if firm.get('website'):
            st.markdown(f"**🌐 Website:** [{firm['website']}]({firm['website']})")
    
    if firm.get('description'):
        st.markdown(f"**📄 About:** {firm['description']}")
    
    # People at this firm
    st.markdown("---")
    st.subheader(f"👥 People at {firm['name']}")
    
    firm_people = get_people_by_firm(firm['name'])
    if firm_people:
        for person in firm_people:
            with st.expander(f"{person['name']} - {person['current_title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    if person.get('email'):
                        st.markdown(f"📧 {person['email']}")
                    if person.get('phone'):
                        st.markdown(f"📱 {person['phone']}")
                    if person.get('education'):
                        st.markdown(f"🎓 {person['education']}")
                
                with col2:
                    if person.get('expertise'):
                        st.markdown(f"🏆 **Expertise:** {person['expertise']}")
                    if person.get('aum_managed'):
                        st.markdown(f"💰 **AUM:** {person['aum_managed']}")
                
                if st.button("View Full Profile", key=f"view_full_{person['id']}"):
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
    
    # Back button
    if st.button("← Back to People"):
        go_to_people()
        st.rerun()
    
    st.header(f"👤 {person['name']}")
    st.subheader(f"{person['current_title']} at {person['current_company_name']}")
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**📍 Location:** {person['location']}")
        if person.get('email'):
            st.markdown(f"**📧 Email:** [{person['email']}](mailto:{person['email']})")
        if person.get('phone'):
            st.markdown(f"**📱 Phone:** {person['phone']}")
        if person.get('linkedin_profile_url'):
            st.markdown(f"**🔗 LinkedIn:** [Profile]({person['linkedin_profile_url']})")
    
    with col2:
        if person.get('education'):
            st.markdown(f"**🎓 Education:** {person['education']}")
        if person.get('expertise'):
            st.markdown(f"**🏆 Expertise:** {person['expertise']}")
        if person.get('aum_managed'):
            st.markdown(f"**💰 AUM Managed:** {person['aum_managed']}")
        if person.get('strategy'):
            st.markdown(f"**📈 Strategy:** {person['strategy']}")
    
    # Employment History
    st.markdown("---")
    st.subheader("💼 Employment History")
    
    employments = get_employments_by_person_id(person['id'])
    if employments:
        # Sort by start date (most recent first)
        sorted_employments = sorted(employments, key=lambda x: x['start_date'], reverse=True)
        
        for emp in sorted_employments:
            end_date_str = emp['end_date'].strftime("%B %Y") if emp['end_date'] else "Present"
            start_date_str = emp['start_date'].strftime("%B %Y")
            
            # Calculate duration
            end_for_calc = emp['end_date'] if emp['end_date'] else date.today()
            duration_days = (end_for_calc - emp['start_date']).days
            duration_years = duration_days / 365.25
            
            if duration_years >= 1:
                duration_str = f"{duration_years:.1f} years"
            else:
                duration_str = f"{duration_days // 30} months"
            
            st.markdown(f"""
            **{emp['title']}** at **{emp['company_name']}**  
            📅 {start_date_str} → {end_date_str} ({duration_str})  
            📍 {emp.get('location', 'Unknown')} • 📈 {emp.get('strategy', 'Unknown')}
            """)
    else:
        st.info("No employment history available.")
    
    # Shared Work History (The OWL-like feature)
    st.markdown("---")
    st.subheader("🤝 Professional Network Connections")
    
    shared_history = get_shared_work_history(person['id'])
    
    if shared_history:
        st.write(f"**Found {len(shared_history)} colleagues who worked at the same companies:**")
        
        # Create a nice table
        network_data = []
        for connection in shared_history:
            network_data.append({
                "Colleague": connection['colleague_name'],
                "Shared Company": connection['shared_company'],
                "Overlap (Years)": connection['overlap_years'],
                "Current Role": f"{connection['colleague_current_title']} at {connection['colleague_current_company']}",
                "Colleague ID": connection['colleague_id']
            })
        
        df_network = pd.DataFrame(network_data)
        
        # Display as interactive table
        st.dataframe(df_network.drop(columns=['Colleague ID']), use_container_width=True)
        
        # Quick access buttons for top connections
        st.write("**Quick Access to Top Connections:**")
        top_connections = shared_history[:5]  # Top 5 connections
        
        cols = st.columns(min(5, len(top_connections)))
        for i, connection in enumerate(top_connections):
            with cols[i]:
                if st.button(f"View {connection['colleague_name'].split()[0]}", 
                           key=f"quick_view_{connection['colleague_id']}",
                           use_container_width=True):
                    go_to_person_details(connection['colleague_id'])
                    st.rerun()
                st.caption(f"{connection['overlap_years']} years together")
        
        # Visualization of network
        if len(shared_history) > 0:
            st.markdown("---")
            st.subheader("📊 Network Visualization")
            
            # Create a simple network chart
            fig = go.Figure()
            
            # Add connections as a bar chart
            companies = [conn['shared_company'] for conn in shared_history]
            overlaps = [conn['overlap_years'] for conn in shared_history]
            colleagues = [conn['colleague_name'] for conn in shared_history]
            
            fig.add_trace(go.Bar(
                x=companies,
                y=overlaps,
                text=[f"{colleague}<br>{overlap}y" for colleague, overlap in zip(colleagues, overlaps)],
                textposition='auto',
                name="Work Overlap"
            ))
            
            fig.update_layout(
                title=f"Professional Network Connections for {person['name']}",
                xaxis_title="Companies",
                yaxis_title="Overlap Years",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No shared work history found with other people in the database.")
        st.write("💡 Add more people who worked at the same companies to see connections!")

# --- Footer ---
st.markdown("---")
st.markdown("### 🌏 Asian Hedge Fund Talent Intelligence Platform")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🤖 AI-Powered Extraction**")
with col2:
    st.markdown("**🤝 Professional Networks**") 
with col3:
    st.markdown("**💾 Persistent Data Storage**")

# Auto-save data on any changes
if st.session_state.get('data_changed', False):
    save_data()
    st.session_state.data_changed = False
