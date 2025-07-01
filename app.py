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

# --- Helper function to safely get string values ---
def safe_get(data, key, default='Unknown'):
    """Safely get a value from dict, ensuring it's not None"""
    value = data.get(key, default)
    return value if value is not None else default

# --- Database Persistence Setup ---
DATA_DIR = Path("hedge_fund_data")
DATA_DIR.mkdir(exist_ok=True)

PEOPLE_FILE = DATA_DIR / "people.json"
FIRMS_FILE = DATA_DIR / "firms.json"
EMPLOYMENTS_FILE = DATA_DIR / "employments.json"
EXTRACTIONS_FILE = DATA_DIR / "extractions.json"

def save_data():
    """Save all data to JSON files with better error handling"""
    try:
        # Ensure directory exists
        DATA_DIR.mkdir(exist_ok=True)
        
        # Save people
        with open(PEOPLE_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.people, f, indent=2, default=str)
        
        # Save firms
        with open(FIRMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.firms, f, indent=2, default=str)
        
        # Save employments
        with open(EMPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.employments, f, indent=2, default=str)
        
        # Save extractions
        if 'all_extractions' in st.session_state:
            with open(EXTRACTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.all_extractions, f, indent=2, default=str)
        
        # Verify files were actually written
        files_saved = []
        if PEOPLE_FILE.exists():
            files_saved.append(f"people.json ({PEOPLE_FILE.stat().st_size} bytes)")
        if FIRMS_FILE.exists():
            files_saved.append(f"firms.json ({FIRMS_FILE.stat().st_size} bytes)")
        if EMPLOYMENTS_FILE.exists():
            files_saved.append(f"employments.json ({EMPLOYMENTS_FILE.stat().st_size} bytes)")
        if EXTRACTIONS_FILE.exists():
            files_saved.append(f"extractions.json ({EXTRACTIONS_FILE.stat().st_size} bytes)")
        
        st.sidebar.success(f"💾 Data saved: {', '.join(files_saved)}")
        return True
        
    except Exception as e:
        st.sidebar.error(f"❌ Save error: {e}")
        st.sidebar.error(f"📁 Attempted to save to: {DATA_DIR.absolute()}")
        return False

def load_data():
    """Load data from JSON files with detailed logging"""
    try:
        people = []
        firms = []
        employments = []
        extractions = []
        
        # Load people
        if PEOPLE_FILE.exists():
            with open(PEOPLE_FILE, 'r', encoding='utf-8') as f:
                people = json.load(f)
            print(f"✅ Loaded {len(people)} people from {PEOPLE_FILE}")
        else:
            print(f"⚠️ No people file found at {PEOPLE_FILE}")
        
        # Load firms
        if FIRMS_FILE.exists():
            with open(FIRMS_FILE, 'r', encoding='utf-8') as f:
                firms = json.load(f)
            print(f"✅ Loaded {len(firms)} firms from {FIRMS_FILE}")
        else:
            print(f"⚠️ No firms file found at {FIRMS_FILE}")
        
        # Load employments
        if EMPLOYMENTS_FILE.exists():
            with open(EMPLOYMENTS_FILE, 'r', encoding='utf-8') as f:
                employments = json.load(f)
                # Convert date strings back to date objects
                for emp in employments:
                    if emp.get('start_date'):
                        emp['start_date'] = datetime.strptime(emp['start_date'], '%Y-%m-%d').date()
                    if emp.get('end_date'):
                        emp['end_date'] = datetime.strptime(emp['end_date'], '%Y-%m-%d').date()
            print(f"✅ Loaded {len(employments)} employments from {EMPLOYMENTS_FILE}")
        else:
            print(f"⚠️ No employments file found at {EMPLOYMENTS_FILE}")
        
        # Load extractions
        if EXTRACTIONS_FILE.exists():
            with open(EXTRACTIONS_FILE, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
            print(f"✅ Loaded {len(extractions)} extractions from {EXTRACTIONS_FILE}")
        else:
            print(f"⚠️ No extractions file found at {EXTRACTIONS_FILE}")
        
        print(f"📁 Data directory: {DATA_DIR.absolute()}")
        
        return people, firms, employments, extractions
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
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
    if 'show_edit_person_modal' not in st.session_state:
        st.session_state.show_edit_person_modal = False
    if 'show_edit_firm_modal' not in st.session_state:
        st.session_state.show_edit_firm_modal = False
    if 'edit_person_data' not in st.session_state:
        st.session_state.edit_person_data = None
    if 'edit_firm_data' not in st.session_state:
        st.session_state.edit_firm_data = None
    if 'pending_updates' not in st.session_state:
        st.session_state.pending_updates = []
    if 'show_update_review' not in st.session_state:
        st.session_state.show_update_review = False

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

def extract_talent_simple(text, model):
    """Crash-proof extraction with minimal UI updates"""
    if not model:
        return []
    
    # Preprocess and clean the text first
    cleaned_text = preprocess_newsletter_text(text)
    
    # Simple size-based processing decision
    max_single_chunk = 15000
    
    if len(cleaned_text) <= max_single_chunk:
        # Single chunk - simple and reliable
        st.info("📄 Processing as single chunk...")
        return extract_single_chunk_safe(cleaned_text, model)
    else:
        # Multi-chunk with crash protection
        st.info(f"📊 Large file detected ({len(cleaned_text):,} chars). Using safe chunking...")
        return extract_multi_chunk_safe(cleaned_text, model, max_single_chunk)

def preprocess_newsletter_text(text):
    """Clean and preprocess newsletter text to remove noise and focus on relevant content"""
    import re
    
    # Show original size
    original_size = len(text)
    
    # Step 1: Remove email headers (From, To, Subject, Sent, etc.)
    email_header_patterns = [
        r'From:\s*.*?\n',
        r'To:\s*.*?\n', 
        r'Sent:\s*.*?\n',
        r'Subject:\s*.*?\n',
        r'Date:\s*.*?\n',
        r'Reply-To:\s*.*?\n',
        r'Return-Path:\s*.*?\n'
    ]
    
    for pattern in email_header_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Step 2: Remove URLs and tracking links
    url_patterns = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',  # Standard URLs
        r'<https?://[^>]+>',  # URLs in angle brackets
        r'urldefense\.proofpoint\.com[^\s]*',  # Proofpoint URLs
        r'pardot\.withintelligence\.com[^\s]*',  # Tracking URLs
        r'jpmorgan\.email\.streetcontxt\.net[^\s]*'  # Email tracking
    ]
    
    for pattern in url_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 3: Remove email disclaimers and legal text
    disclaimer_patterns = [
        r'This section contains materials produced by third parties.*?(?=\n\n|\Z)',
        r'This message is confidential and subject to terms.*?(?=\n\n|\Z)',
        r'Important Reminder: JPMorgan Chase will never send emails.*?(?=\n\n|\Z)',
        r'Although this transmission and any links.*?(?=\n\n|\Z)',
        r'©.*?All rights reserved.*?(?=\n\n|\Z)',
        r'Unsubscribe.*?(?=\n\n|\Z)',
        r'Privacy Policy.*?(?=\n\n|\Z)',
        r'Update email preferences.*?(?=\n\n|\Z)',
        r'Not seeing what you expected\?.*?(?=\n\n|\Z)',
        r'Log in to my account.*?(?=\n\n|\Z)'
    ]
    
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 4: Remove HTML artifacts and email formatting
    html_patterns = [
        r'<[^>]+>',  # HTML tags
        r'&[a-zA-Z0-9#]+;',  # HTML entities
        r'\[cid:[^\]]+\]',  # Email embedded images
        r'________________________________+',  # Email separators
        r'\*\s*\|.*?\|\s*\*',  # Email table formatting
    ]
    
    for pattern in html_patterns:
        text = re.sub(pattern, '', text)
    
    # Step 5: Remove tracking and navigation elements
    navigation_patterns = [
        r'View article\s*View mandate\s*',
        r'View investor\s*View mandate\s*',
        r'Asset class:.*?(?=\n)',
        r'Region:.*?(?=\n)',
        r'Strategy:.*?(?=\n)',
        r'Contact us\s*',
        r'SEE ALL ARTICLES.*?(?=\n)',
        r'CAPITAL ADVISORY GROUP.*?(?=\n)',
    ]
    
    for pattern in navigation_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 6: Clean up excessive whitespace and formatting
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove lines with only special characters or whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Keep line if it has meaningful content (letters and useful words)
        if re.search(r'[a-zA-Z].*[a-zA-Z]', line) and len(line.strip()) > 5:
            # Clean up the line
            line = re.sub(r'\s+', ' ', line.strip())  # Normalize whitespace
            if line:
                cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Step 7: Focus on hedge fund relevant content
    # Look for common hedge fund keywords and keep paragraphs containing them
    hf_keywords = [
        'hedge fund', 'portfolio manager', 'pm', 'cio', 'chief investment officer',
        'managing director', 'md', 'vice president', 'vp', 'analyst', 'trader',
        'fund launch', 'fund debut', 'joins', 'moves', 'promotes', 'appoints',
        'former', 'ex-', 'launches', 'capital management', 'partners', 'advisors',
        'assets under management', 'aum', 'long/short', 'equity', 'credit',
        'quantitative', 'macro', 'multi-strategy', 'arbitrage'
    ]
    
    # Split into paragraphs and keep relevant ones
    paragraphs = text.split('\n\n')
    relevant_paragraphs = []
    
    for para in paragraphs:
        para_lower = para.lower()
        if any(keyword in para_lower for keyword in hf_keywords):
            relevant_paragraphs.append(para)
        elif len(para) > 100 and ('capital' in para_lower or 'management' in para_lower):
            # Keep longer paragraphs that might be relevant
            relevant_paragraphs.append(para)
    
    # If we didn't find enough relevant content, keep more of the original
    if len(relevant_paragraphs) < 3:
        relevant_paragraphs = paragraphs[:10]  # Keep first 10 paragraphs as fallback
    
    cleaned_text = '\n\n'.join(relevant_paragraphs)
    
    # Final cleanup
    cleaned_text = cleaned_text.strip()
    
    # Show cleaning results
    final_size = len(cleaned_text)
    reduction_pct = ((original_size - final_size) / original_size) * 100 if original_size > 0 else 0
    
    st.info(f"📝 **Text Preprocessing Complete**")
    st.write(f"• **Original size**: {original_size:,} characters")
    st.write(f"• **Cleaned size**: {final_size:,} characters") 
    st.write(f"• **Reduction**: {reduction_pct:.1f}% noise removed")
    st.write(f"• **Relevant paragraphs**: {len(relevant_paragraphs)} found")
    
    return cleaned_text

def find_similar_person(extracted_person):
    """Find existing person that might match the extracted data"""
    extracted_name = safe_get(extracted_person, 'name', '').lower().strip()
    if not extracted_name:
        return None
    
    # Try exact name match first
    for person in st.session_state.people:
        existing_name = safe_get(person, 'name', '').lower().strip()
        if existing_name == extracted_name:
            return person
    
    # Try fuzzy matching (first name + last name)
    extracted_parts = extracted_name.split()
    if len(extracted_parts) >= 2:
        extracted_first = extracted_parts[0]
        extracted_last = extracted_parts[-1]
        
        for person in st.session_state.people:
            existing_name = safe_get(person, 'name', '').lower().strip()
            existing_parts = existing_name.split()
            if len(existing_parts) >= 2:
                existing_first = existing_parts[0]
                existing_last = existing_parts[-1]
                
                # Match if first and last names are the same
                if extracted_first == existing_first and extracted_last == existing_last:
                    return person
    
    return None

def detect_updates_needed(existing_person, extracted_person):
    """Compare existing person data with extracted data to find potential updates"""
    updates = {}
    
    # Check company change
    existing_company = safe_get(existing_person, 'current_company_name')
    extracted_company = safe_get(extracted_person, 'company')
    if extracted_company != 'Unknown' and existing_company != extracted_company:
        updates['company'] = {
            'field': 'current_company_name',
            'current': existing_company,
            'proposed': extracted_company,
            'reason': 'Company change detected'
        }
    
    # Check title change
    existing_title = safe_get(existing_person, 'current_title')
    extracted_title = safe_get(extracted_person, 'title')
    if extracted_title != 'Unknown' and existing_title != extracted_title:
        updates['title'] = {
            'field': 'current_title',
            'current': existing_title,
            'proposed': extracted_title,
            'reason': 'Title change detected'
        }
    
    # Check location change
    existing_location = safe_get(existing_person, 'location')
    extracted_location = safe_get(extracted_person, 'location')
    if extracted_location != 'Unknown' and existing_location != extracted_location:
        updates['location'] = {
            'field': 'location',
            'current': existing_location,
            'proposed': extracted_location,
            'reason': 'Location change detected'
        }
    
    return updates

def process_extractions_with_update_detection(extractions):
    """Process extractions and detect potential updates for existing people"""
    new_people = []
    pending_updates = []
    
    for extracted in extractions:
        existing_person = find_similar_person(extracted)
        
        if existing_person:
            # Person exists - check for updates
            updates = detect_updates_needed(existing_person, extracted)
            
            if updates:
                # Found potential updates
                pending_updates.append({
                    'id': str(uuid.uuid4()),
                    'person_id': existing_person['id'],
                    'person_name': safe_get(existing_person, 'name'),
                    'updates': updates,
                    'extracted_data': extracted,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                # No updates needed
                st.info(f"✅ {safe_get(extracted, 'name')} - No updates needed")
        else:
            # New person
            new_people.append(extracted)
    
    return new_people, pending_updates

# --- Helper Functions ---
def get_person_by_id(person_id):
    return next((p for p in st.session_state.people if p['id'] == person_id), None)

def get_firm_by_id(firm_id):
    return next((f for f in st.session_state.firms if f['id'] == firm_id), None)

def get_firm_by_name(firm_name):
    return next((f for f in st.session_state.firms if f['name'] == firm_name), None)

def get_people_by_firm(firm_name):
    return [p for p in st.session_state.people if safe_get(p, 'current_company_name') == firm_name]

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
                            "colleague_name": safe_get(other_person, 'name'),
                            "colleague_id": other_person['id'],
                            "shared_company": person_emp['company_name'],
                            "colleague_current_company": safe_get(other_person, 'current_company_name'),
                            "colleague_current_title": safe_get(other_person, 'current_title'),
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
        
        # Show simple rate limit info
        if model:
            model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
            
            if '1.5-pro' in model_id:
                st.info("🧠 Gemini 1.5 Pro: Slowest but most thorough (40s between chunks)")
            elif '2.5-flash' in model_id:
                st.info("🌟 Gemini 2.5 Flash: Latest model (10s between chunks)")
            else:
                st.info("⚡ Gemini Flash: Fast and reliable (10s between chunks)")
        
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
        
        # Show simple processing info
        if newsletter_text and len(newsletter_text) > 15000:
            char_count = len(newsletter_text)
            chunks_needed = max(1, char_count // 15000)
            st.info(f"Large file: {char_count:,} chars → {chunks_needed} chunks → ~{chunks_needed * 2} minutes estimated")
        
        # Show preprocessing preview
        if newsletter_text:
            with st.expander("🔍 Preview Text Preprocessing"):
                st.write("**Raw text preview (first 500 chars):**")
                st.code(newsletter_text[:500] + "..." if len(newsletter_text) > 500 else newsletter_text)
                
                if st.button("🧹 Test Preprocessing", key="test_preprocess"):
                    cleaned = preprocess_newsletter_text(newsletter_text)
                    st.write("**Cleaned text preview (first 500 chars):**")
                    st.code(cleaned[:500] + "..." if len(cleaned) > 500 else cleaned)

        # Extract button - simplified and crash-proof
        if st.button("🚀 Extract Talent", use_container_width=True):
            if not newsletter_text.strip():
                st.error("Please provide newsletter content")
            elif not model:
                st.error("Please provide API key")
            else:
                char_count = len(newsletter_text)
                st.info(f"📊 Processing {char_count:,} characters...")
                
                # Simplified size checking - no complex confirmations
                if char_count > 150000:  # 150KB hard limit
                    st.error(f"❌ File too large: {char_count:,} characters. Maximum: 150,000")
                    st.info("💡 **Try**: Split into smaller files or copy/paste key sections")
                else:
                    # Show warning for large files but proceed automatically
                    if char_count > 50000:
                        chunks_needed = max(1, char_count // 15000)
                        est_minutes = chunks_needed * 2
                        st.warning(f"⚠️ **Large file**: {char_count:,} chars → ~{est_minutes} min processing time")
                    
                    # Always proceed with processing
                    try:
                        st.info("🤖 **Starting extraction...**")
                        
                        # Show processing status
                        with st.status("Processing newsletter...", expanded=True) as status:
                            st.write("🔄 Initializing extraction...")
                            extractions = extract_talent_simple(newsletter_text, model)
                            
                            if extractions:
                                st.write(f"✅ Found {len(extractions)} people!")
                                status.update(label="✅ Extraction complete!", state="complete")
                            else:
                                st.write("⚠️ No people found")
                                status.update(label="⚠️ No results found", state="complete")
                        
                        if extractions:
                            # Add metadata
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for ext in extractions:
                                ext['timestamp'] = timestamp
                                ext['id'] = str(uuid.uuid4())
                            
                            # Process extractions and detect updates
                            st.info("🔍 Checking for existing people and potential updates...")
                            new_people, pending_updates = process_extractions_with_update_detection(extractions)
                            
                            # Save new extractions
                            st.session_state.all_extractions.extend(extractions)
                            
                            # Add pending updates to session state
                            if pending_updates:
                                st.session_state.pending_updates.extend(pending_updates)
                                st.session_state.show_update_review = True
                            
                            # Save results
                            if save_data():
                                st.success(f"🎉 **Extraction Complete!**")
                            else:
                                st.error("⚠️ Extraction successful but save failed!")
                            
                            # Show summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("👥 New People", len(new_people))
                            with col2:
                                st.metric("🔄 Pending Updates", len(pending_updates))
                            with col3:
                                st.metric("🏢 Total Extracted", len(extractions))
                            
                            # Show what happened
                            if new_people:
                                with st.expander(f"📋 {len(new_people)} New People Found"):
                                    for person in new_people:
                                        st.write(f"• **{person['name']}** → {person['company']}")
                            
                            if pending_updates:
                                st.warning(f"⚠️ Found {len(pending_updates)} potential updates for existing people!")
                                st.info("👆 **Review updates in the sidebar before they're applied**")
                            
                        else:
                            st.warning("⚠️ No people found. Try a different model or check content.")
                            
                    except Exception as e:
                        st.error(f"💥 **Extraction failed**: {str(e)}")
                        st.info("**Try**: Different model, smaller file, or copy/paste instead")
        
        # Quick test - simplified
        if st.button("🧪 Test with Sample", use_container_width=True):
            sample = """
            From: Newsletter <newsletter@hedgefund.com>
            Sent: Friday, June 27, 2025 4:56 PM
            Subject: Daily Hedge Fund News
            
            Goldman Sachs veteran John Smith joins Citadel Asia as Managing Director in Hong Kong.
            Former JPMorgan portfolio manager Lisa Chen launches her own hedge fund, Dragon Capital, 
            focusing on Asian equities. Millennium Partners promotes Alex Wang to head of quant trading 
            in Singapore. Sarah Kim moves from Bridgewater to become CIO at newly formed Tiger Asia.
            
            This message is confidential. Please visit https://tracking.example.com for more.
            Unsubscribe: https://unsubscribe.example.com
            """
            
            try:
                st.info("Testing preprocessing + extraction...")
                
                # Show preprocessing in action
                with st.expander("📝 Preprocessing Results", expanded=True):
                    cleaned_sample = preprocess_newsletter_text(sample)
                
                # Then extract
                extractions = extract_single_chunk_safe(cleaned_sample, model)
                st.write(f"**Found {len(extractions)} people:**")
                for ext in extractions:
                    st.write(f"• {ext.get('name', 'Unknown')} → {ext.get('company', 'Unknown')}")
                    
            except Exception as e:
                st.error(f"Test failed: {e}")
    
    # Show recent extractions
    if st.session_state.all_extractions:
        st.markdown("---")
        st.subheader("📊 Recent Extractions")
        st.metric("Total Extracted", len(st.session_state.all_extractions))
        
        # Add people from extractions with safe defaults
        if st.button("📥 Import New People Only", use_container_width=True):
            # Only import extractions that don't have existing people
            added_count = 0
            skipped_existing = 0
            
            for ext in st.session_state.all_extractions:
                existing_person = find_similar_person(ext)
                
                if existing_person:
                    skipped_existing += 1
                    continue
                
                # Check if person already exists in current database
                existing = any(safe_get(p, 'name', '').lower() == safe_get(ext, 'name', '').lower() 
                             for p in st.session_state.people)
                
                if not existing and ext.get('name') and ext.get('company'):
                    # Add person with safe defaults
                    new_person_id = str(uuid.uuid4())
                    st.session_state.people.append({
                        "id": new_person_id,
                        "name": ext.get('name', 'Unknown'),
                        "current_title": ext.get('title', 'Unknown'),
                        "current_company_name": ext.get('company', 'Unknown'),
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
                    if not get_firm_by_name(ext.get('company', '')):
                        st.session_state.firms.append({
                            "id": str(uuid.uuid4()),
                            "name": ext.get('company', 'Unknown'),
                            "location": ext.get('location', 'Unknown'),
                            "headquarters": "Unknown",
                            "aum": "Unknown",
                            "founded": None,
                            "strategy": "Hedge Fund",
                            "website": "",
                            "description": f"Hedge fund - extracted from newsletter"
                        })
                    
                    # Add employment with safe defaults
                    st.session_state.employments.append({
                        "id": str(uuid.uuid4()),
                        "person_id": new_person_id,
                        "company_name": ext.get('company', 'Unknown'),
                        "title": ext.get('title', 'Unknown'),
                        "start_date": date.today(),
                        "end_date": None,
                        "location": ext.get('location', 'Unknown'),
                        "strategy": "Unknown"
                    })
                    
                    added_count += 1
            
            save_data()  # Save changes
            if skipped_existing > 0:
                st.success(f"✅ Added {added_count} new people! Skipped {skipped_existing} existing people.")
                st.info("💡 Use 'Review Updates' to handle existing people changes")
            else:
                st.success(f"✅ Added {added_count} new people to database!")
            st.rerun()

    # Show pending updates for review
    if st.session_state.pending_updates:
        st.markdown("---")
        st.subheader("🔄 Review Updates")
        st.warning(f"Found {len(st.session_state.pending_updates)} potential updates")
        
        if st.button("📝 Review & Approve Updates", use_container_width=True):
            st.session_state.show_update_review = True
            st.rerun()
        
        # Show quick preview
        with st.expander("👀 Quick Preview"):
            for update in st.session_state.pending_updates[:3]:
                st.write(f"**{update['person_name']}**: {len(update['updates'])} potential changes")
            if len(st.session_state.pending_updates) > 3:
                st.write(f"... and {len(st.session_state.pending_updates) - 3} more")

    elif not GENAI_AVAILABLE:
        st.error("Please install: pip install google-generativeai")

# --- UPDATE REVIEW MODAL ---
if st.session_state.show_update_review and st.session_state.pending_updates:
    st.markdown("---")
    st.header("🔄 Review Potential Updates")
    st.info("AI detected changes for existing people. Please review and approve updates.")
    
    # Process each pending update
    updates_to_remove = []
    
    for i, pending_update in enumerate(st.session_state.pending_updates):
        person_name = pending_update['person_name']
        updates = pending_update['updates']
        
        st.markdown(f"### 👤 {person_name}")
        st.caption(f"Detected on: {pending_update['timestamp']}")
        
        # Show each potential update
        approved_updates = {}
        
        with st.container():
            for update_key, update_info in updates.items():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"**{update_info['reason']}**")
                    st.caption(f"Field: {update_key.title()}")
                
                with col2:
                    st.write("**Current:**")
                    st.code(update_info['current'])
                
                with col3:
                    st.write("**Proposed:**")
                    st.code(update_info['proposed'])
                
                with col4:
                    # Approval checkbox
                    approve_key = f"approve_{pending_update['id']}_{update_key}"
                    if st.checkbox("✅ Approve", key=approve_key):
                        approved_updates[update_key] = update_info
        
        # Action buttons for this person
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"✅ Apply Approved Changes", key=f"apply_{pending_update['id']}", use_container_width=True):
                if approved_updates:
                    # Apply approved updates
                    person_id = pending_update['person_id']
                    person = get_person_by_id(person_id)
                    
                    if person:
                        changes_made = []
                        
                        # Apply each approved update
                        for update_key, update_info in approved_updates.items():
                            old_value = person[update_info['field']]
                            person[update_info['field']] = update_info['proposed']
                            changes_made.append(f"{update_key}: {old_value} → {update_info['proposed']}")
                            
                            # If company changed, add new employment record
                            if update_info['field'] == 'current_company_name':
                                st.session_state.employments.append({
                                    "id": str(uuid.uuid4()),
                                    "person_id": person_id,
                                    "company_name": update_info['proposed'],
                                    "title": safe_get(person, 'current_title'),
                                    "start_date": date.today(),
                                    "end_date": None,
                                    "location": safe_get(person, 'location'),
                                    "strategy": "Unknown"
                                })
                                
                                # End previous employment
                                for emp in st.session_state.employments:
                                    if (emp['person_id'] == person_id and 
                                        emp['company_name'] == old_value and 
                                        emp.get('end_date') is None and
                                        emp['id'] != st.session_state.employments[-1]['id']):
                                        emp['end_date'] = date.today()
                                        break
                        
                        # Update the person in the main list
                        for j, p in enumerate(st.session_state.people):
                            if p['id'] == person_id:
                                st.session_state.people[j] = person
                                break
                        
                        save_data()
                        st.success(f"✅ Updated {person_name}: {', '.join(changes_made)}")
                        updates_to_remove.append(i)
                    
                else:
                    st.warning("No changes approved for this person")
        
        with col2:
            if st.button(f"❌ Reject All", key=f"reject_{pending_update['id']}", use_container_width=True):
                st.info(f"Rejected all updates for {person_name}")
                updates_to_remove.append(i)
        
        with col3:
            if st.button(f"👁️ View Profile", key=f"view_{pending_update['id']}", use_container_width=True):
                person_id = pending_update['person_id']
                go_to_person_details(person_id)
                st.rerun()
        
        st.markdown("---")
    
    # Remove processed updates
    if updates_to_remove:
        for i in sorted(updates_to_remove, reverse=True):
            del st.session_state.pending_updates[i]
        save_data()
        st.rerun()
    
    # Close button
    if st.button("✅ Done Reviewing", use_container_width=True):
        st.session_state.show_update_review = False
        st.rerun()

elif not GENAI_AVAILABLE:
    st.error("Please install: pip install google-generativeai")

# --- MAIN CONTENT AREA ---
st.title("🏢 Asian Hedge Fund Talent Map")
st.markdown("### Professional network mapping for Asia's hedge fund industry")

# Data persistence status indicator
with st.container():
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown("**📊 Database Status:**")
    
    with col2:
        # Check if data files exist and show status
        if PEOPLE_FILE.exists() or FIRMS_FILE.exists():
            last_modified = max(
                PEOPLE_FILE.stat().st_mtime if PEOPLE_FILE.exists() else 0,
                FIRMS_FILE.stat().st_mtime if FIRMS_FILE.exists() else 0
            )
            last_save = datetime.fromtimestamp(last_modified).strftime("%H:%M:%S")
            st.success(f"✅ Saved at {last_save}")
        else:
            st.warning("⚠️ No saved data")
    
    with col3:
        if st.button("💾 Force Save"):
            if save_data():
                st.success("✅ Saved!")
            st.rerun()
    
    with col4:
        # Auto-save every time data changes
        total_items = len(st.session_state.people) + len(st.session_state.firms) + len(st.session_state.all_extractions)
        if 'last_saved_count' not in st.session_state:
            st.session_state.last_saved_count = total_items
        
        if total_items != st.session_state.last_saved_count:
            st.info("💾 Auto-saving...")
            save_data()
            st.session_state.last_saved_count = total_items
        
        st.caption(f"📁 {DATA_DIR.absolute()}")

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

# --- EDIT PERSON MODAL ---
if st.session_state.show_edit_person_modal and st.session_state.edit_person_data:
    st.markdown("---")
    st.subheader(f"✏️ Edit {safe_get(st.session_state.edit_person_data, 'name', 'Person')}")
    
    person_data = st.session_state.edit_person_data
    
    with st.form("edit_person_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", value=safe_get(person_data, 'name'))
            title = st.text_input("Current Title*", value=safe_get(person_data, 'current_title'))
            current_company = safe_get(person_data, 'current_company_name')
            company_options = [""] + [f['name'] for f in st.session_state.firms]
            company_index = 0
            if current_company and current_company in company_options:
                company_index = company_options.index(current_company)
            company = st.selectbox("Current Company*", options=company_options, index=company_index)
            
            current_location = safe_get(person_data, 'location')
            location_options = ["", "Hong Kong", "Singapore", "Tokyo", "Seoul", "Mumbai", "Shanghai", "Beijing", "Taipei"]
            location_index = 0
            if current_location and current_location in location_options:
                location_index = location_options.index(current_location)
            location = st.selectbox("Location*", options=location_options, index=location_index)
        
        with col2:
            email = st.text_input("Email", value=safe_get(person_data, 'email'))
            phone = st.text_input("Phone", value=safe_get(person_data, 'phone'))
            linkedin = st.text_input("LinkedIn URL", value=safe_get(person_data, 'linkedin_profile_url'))
            education = st.text_input("Education", value=safe_get(person_data, 'education'))
        
        col3, col4 = st.columns(2)
        with col3:
            expertise = st.text_input("Expertise", value=safe_get(person_data, 'expertise'))
            aum = st.text_input("AUM Managed", value=safe_get(person_data, 'aum_managed'))
        
        with col4:
            current_strategy = safe_get(person_data, 'strategy')
            strategy_options = ["", "Equity Long/Short", "Multi-Strategy", "Quantitative", "Macro", "Credit"]
            strategy_index = 0
            if current_strategy and current_strategy in strategy_options:
                strategy_index = strategy_options.index(current_strategy)
            strategy = st.selectbox("Investment Strategy", options=strategy_options, index=strategy_index)
        
        col_save, col_cancel, col_delete = st.columns(3)
        
        with col_save:
            if st.form_submit_button("💾 Save Changes", use_container_width=True):
                if name and title and company and location:
                    # Update person data
                    person_data.update({
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
                    
                    # Find and update the person in the main list
                    for i, p in enumerate(st.session_state.people):
                        if p['id'] == person_data['id']:
                            st.session_state.people[i] = person_data
                            break
                    
                    save_data()
                    st.success(f"✅ Updated {name}!")
                    st.session_state.show_edit_person_modal = False
                    st.session_state.edit_person_data = None
                    st.rerun()
                else:
                    st.error("Please fill required fields (*)")
        
        with col_cancel:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.show_edit_person_modal = False
                st.session_state.edit_person_data = None
                st.rerun()
        
        with col_delete:
            if st.form_submit_button("🗑️ Delete Person", use_container_width=True):
                # Remove person and related data
                person_id = person_data['id']
                st.session_state.people = [p for p in st.session_state.people if p['id'] != person_id]
                st.session_state.employments = [e for e in st.session_state.employments if e['person_id'] != person_id]
                
                save_data()
                st.success("✅ Person deleted!")
                st.session_state.show_edit_person_modal = False
                st.session_state.edit_person_data = None
                st.rerun()

# --- EDIT FIRM MODAL ---
if st.session_state.show_edit_firm_modal and st.session_state.edit_firm_data:
    st.markdown("---")
    st.subheader(f"✏️ Edit {safe_get(st.session_state.edit_firm_data, 'name', 'Firm')}")
    
    firm_data = st.session_state.edit_firm_data
    
    with st.form("edit_firm_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            firm_name = st.text_input("Firm Name*", value=safe_get(firm_data, 'name'))
            current_location = safe_get(firm_data, 'location')
            location_options = ["", "Hong Kong", "Singapore", "Tokyo", "Seoul", "Mumbai", "Shanghai", "Beijing", "Taipei"]
            location_index = 0
            if current_location and current_location in location_options:
                location_index = location_options.index(current_location)
            location = st.selectbox("Location*", options=location_options, index=location_index)
            
            headquarters = st.text_input("Headquarters", value=safe_get(firm_data, 'headquarters'))
            aum = st.text_input("AUM", value=safe_get(firm_data, 'aum'))
            
        with col2:
            current_strategy = safe_get(firm_data, 'strategy')
            strategy_options = ["", "Long/Short Equity", "Multi-Strategy", "Quantitative", "Macro", "Event Driven"]
            strategy_index = 0
            if current_strategy and current_strategy in strategy_options:
                strategy_index = strategy_options.index(current_strategy)
            strategy = st.selectbox("Strategy", options=strategy_options, index=strategy_index)
            
            founded = st.number_input("Founded", min_value=1900, max_value=2025, 
                                    value=firm_data.get('founded', 2000) if firm_data.get('founded') else 2000)
            website = st.text_input("Website", value=safe_get(firm_data, 'website'))
        
        description = st.text_area("Description", value=safe_get(firm_data, 'description'))
        
        col_save, col_cancel, col_delete = st.columns(3)
        
        with col_save:
            if st.form_submit_button("💾 Save Changes", use_container_width=True):
                if firm_name and location:
                    # Update firm data
                    old_name = safe_get(firm_data, 'name')
                    firm_data.update({
                        "name": firm_name,
                        "location": location,
                        "headquarters": headquarters,
                        "aum": aum,
                        "founded": founded if founded > 1900 else None,
                        "strategy": strategy,
                        "website": website,
                        "description": description
                    })
                    
                    # Find and update the firm in the main list
                    for i, f in enumerate(st.session_state.firms):
                        if f['id'] == firm_data['id']:
                            st.session_state.firms[i] = firm_data
                            break
                    
                    # Update people's company names if firm name changed
                    if old_name != firm_name:
                        for person in st.session_state.people:
                            if safe_get(person, 'current_company_name') == old_name:
                                person['current_company_name'] = firm_name
                    
                    save_data()
                    st.success(f"✅ Updated {firm_name}!")
                    st.session_state.show_edit_firm_modal = False
                    st.session_state.edit_firm_data = None
                    st.rerun()
                else:
                    st.error("Please fill Firm Name and Location")
        
        with col_cancel:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.show_edit_firm_modal = False
                st.session_state.edit_firm_data = None
                st.rerun()
        
        with col_delete:
            if st.form_submit_button("🗑️ Delete Firm", use_container_width=True):
                # Remove firm and update related data
                firm_id = firm_data['id']
                firm_name = safe_get(firm_data, 'name')
                
                st.session_state.firms = [f for f in st.session_state.firms if f['id'] != firm_id]
                
                # Update people to remove company reference
                for person in st.session_state.people:
                    if safe_get(person, 'current_company_name') == firm_name:
                        person['current_company_name'] = 'Unknown'
                
                save_data()
                st.success("✅ Firm deleted!")
                st.session_state.show_edit_firm_modal = False
                st.session_state.edit_firm_data = None
                st.rerun()

# --- FIRMS VIEW ---
if st.session_state.current_view == 'firms':
    st.markdown("---")
    st.header("🏢 Hedge Funds in Asia")
    
    if not st.session_state.firms:
        st.info("No firms added yet. Use 'Add Firm' button above.")
    else:
        # Display firms in compact cards
        for firm in st.session_state.firms:
            people_count = len(get_people_by_firm(safe_get(firm, 'name')))
            
            # Compact card design
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**🏢 {safe_get(firm, 'name')}**")
                    st.caption(f"{safe_get(firm, 'strategy')} • {safe_get(firm, 'location')}")
                
                with col2:
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("💰 AUM", safe_get(firm, 'aum'), label_visibility="collapsed")
                    with col2b:
                        st.metric("👥 People", people_count, label_visibility="collapsed")
                
                with col3:
                    col3a, col3b = st.columns(2)
                    with col3a:
                        if st.button("👁️", key=f"view_firm_{firm['id']}", help="View Details"):
                            go_to_firm_details(firm['id'])
                            st.rerun()
                    with col3b:
                        if st.button("✏️", key=f"edit_firm_{firm['id']}", help="Edit Firm"):
                            st.session_state.edit_firm_data = firm
                            st.session_state.show_edit_firm_modal = True
                            st.rerun()
                
                st.markdown("---")

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
            # Filter out None values before sorting
            locations = ["All"] + sorted(list(set(safe_get(p, 'location') for p in st.session_state.people if safe_get(p, 'location') != 'Unknown')))
            location_filter = st.selectbox("Filter by Location", locations)
        with col2:
            # Filter out None values before sorting
            companies = ["All"] + sorted(list(set(safe_get(p, 'current_company_name') for p in st.session_state.people if safe_get(p, 'current_company_name') != 'Unknown')))
            company_filter = st.selectbox("Filter by Company", companies)
        with col3:
            search_term = st.text_input("Search by Name", placeholder="Enter name...")
        
        # Apply filters
        filtered_people = st.session_state.people
        if location_filter != "All":
            filtered_people = [p for p in filtered_people if safe_get(p, 'location') == location_filter]
        if company_filter != "All":
            filtered_people = [p for p in filtered_people if safe_get(p, 'current_company_name') == company_filter]
        if search_term:
            filtered_people = [p for p in filtered_people if search_term.lower() in safe_get(p, 'name').lower()]
        
        # Display people in compact cards
        st.write(f"**Showing {len(filtered_people)} people**")
        
        for person in filtered_people:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**👤 {safe_get(person, 'name')}**")
                    st.caption(f"{safe_get(person, 'current_title')} • {safe_get(person, 'current_company_name')}")
                
                with col2:
                    col2a, col2b = st.columns(2)
                    with col2a:
                        location = safe_get(person, 'location')
                        if len(location) > 10:
                            location = location[:10] + "..."
                        st.metric("📍", location, label_visibility="collapsed")
                    with col2b:
                        aum = safe_get(person, 'aum_managed')
                        if len(aum) > 8:
                            aum = aum[:8] + "..."
                        st.metric("💰", aum, label_visibility="collapsed")
                
                with col3:
                    col3a, col3b = st.columns(2)
                    with col3a:
                        if st.button("👁️", key=f"view_person_{person['id']}", help="View Profile"):
                            go_to_person_details(person['id'])
                            st.rerun()
                    with col3b:
                        if st.button("✏️", key=f"edit_person_{person['id']}", help="Edit Person"):
                            st.session_state.edit_person_data = person
                            st.session_state.show_edit_person_modal = True
                            st.rerun()
                
                st.markdown("---")

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
        st.header(f"🏢 {safe_get(firm, 'name')}")
        st.markdown(f"**{safe_get(firm, 'strategy')} Hedge Fund** • {safe_get(firm, 'location')}")
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("← Back"):
                go_to_firms()
                st.rerun()
        with col2b:
            if st.button("✏️ Edit"):
                st.session_state.edit_firm_data = firm
                st.session_state.show_edit_firm_modal = True
                st.rerun()
    
    # Firm details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Assets Under Management", safe_get(firm, 'aum'))
    with col2:
        st.metric("Founded", safe_get(firm, 'founded'))
    with col3:
        people_count = len(get_people_by_firm(safe_get(firm, 'name')))
        st.metric("Total People", people_count)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**📍 Location:** {safe_get(firm, 'location')}")
        st.markdown(f"**🏛️ Headquarters:** {safe_get(firm, 'headquarters')}")
    with col2:
        st.markdown(f"**📈 Strategy:** {safe_get(firm, 'strategy')}")
        website = safe_get(firm, 'website')
        if website:
            st.markdown(f"**🌐 Website:** [{website}]({website})")
    
    description = safe_get(firm, 'description')
    if description:
        st.markdown(f"**📄 About:** {description}")
    
    # People at this firm
    st.markdown("---")
    st.subheader(f"👥 People at {safe_get(firm, 'name')}")
    
    firm_people = get_people_by_firm(safe_get(firm, 'name'))
    if firm_people:
        for person in firm_people:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**👤 {safe_get(person, 'name')}**")
                    st.caption(safe_get(person, 'current_title'))
                
                with col2:
                    contact_items = []
                    email = safe_get(person, 'email')
                    if email:
                        contact_items.append(f"📧 {email}")
                    aum = safe_get(person, 'aum_managed')
                    if aum:
                        contact_items.append(f"💰 {aum}")
                    
                    if contact_items:
                        st.caption(" • ".join(contact_items[:2]))
                    else:
                        st.caption("No contact info")
                
                with col3:
                    if st.button("👁️ Profile", key=f"view_full_{person['id']}", use_container_width=True):
                        go_to_person_details(person['id'])
                        st.rerun()
                
                st.markdown("---")
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
        st.header(f"👤 {safe_get(person, 'name')}")
        st.subheader(f"{safe_get(person, 'current_title')} at {safe_get(person, 'current_company_name')}")
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("← Back"):
                go_to_people()
                st.rerun()
        with col2b:
            if st.button("✏️ Edit"):
                st.session_state.edit_person_data = person
                st.session_state.show_edit_person_modal = True
                st.rerun()
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**📍 Location:** {safe_get(person, 'location')}")
        email = safe_get(person, 'email')
        if email:
            st.markdown(f"**📧 Email:** [{email}](mailto:{email})")
        phone = safe_get(person, 'phone')
        if phone:
            st.markdown(f"**📱 Phone:** {phone}")
        linkedin = safe_get(person, 'linkedin_profile_url')
        if linkedin:
            st.markdown(f"**🔗 LinkedIn:** [Profile]({linkedin})")
    
    with col2:
        education = safe_get(person, 'education')
        if education:
            st.markdown(f"**🎓 Education:** {education}")
        expertise = safe_get(person, 'expertise')
        if expertise:
            st.markdown(f"**🏆 Expertise:** {expertise}")
        aum = safe_get(person, 'aum_managed')
        if aum:
            st.markdown(f"**💰 AUM Managed:** {aum}")
        strategy = safe_get(person, 'strategy')
        if strategy:
            st.markdown(f"**📈 Strategy:** {strategy}")
    
    # Employment History
    st.markdown("---")
    st.subheader("💼 Employment History")
    
    employments = get_employments_by_person_id(person['id'])
    if employments:
        # Sort by start date (most recent first) - handle None values
        sorted_employments = sorted(
            [emp for emp in employments if emp.get('start_date')], 
            key=lambda x: x['start_date'], 
            reverse=True
        )
        
        for emp in sorted_employments:
            end_date_str = emp['end_date'].strftime("%B %Y") if emp.get('end_date') else "Present"
            start_date_str = emp['start_date'].strftime("%B %Y") if emp.get('start_date') else "Unknown"
            
            # Calculate duration safely
            if emp.get('start_date'):
                end_for_calc = emp['end_date'] if emp.get('end_date') else date.today()
                duration_days = (end_for_calc - emp['start_date']).days
                duration_years = duration_days / 365.25
                
                if duration_years >= 1:
                    duration_str = f"{duration_years:.1f} years"
                else:
                    duration_str = f"{max(1, duration_days // 30)} months"
            else:
                duration_str = "Unknown duration"
            
            st.markdown(f"""
            **{safe_get(emp, 'title')}** at **{safe_get(emp, 'company_name')}**  
            📅 {start_date_str} → {end_date_str} ({duration_str})  
            📍 {safe_get(emp, 'location')} • 📈 {safe_get(emp, 'strategy')}
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
                title=f"Professional Network Connections for {safe_get(person, 'name')}",
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

# Automatic data saving and backup
if st.button("📥 Export Database Backup", use_container_width=True):
    export_data = {
        "people": st.session_state.people,
        "firms": st.session_state.firms,
        "employments": st.session_state.employments,
        "extractions": st.session_state.all_extractions,
        "export_timestamp": datetime.now().isoformat(),
        "total_records": len(st.session_state.people) + len(st.session_state.firms)
    }
    
    export_json = json.dumps(export_data, indent=2, default=str)
    st.download_button(
        "💾 Download Complete Database",
        export_json,
        f"hedge_fund_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json",
        use_container_width=True
    )

# Force save data periodically
current_time = datetime.now()
if 'last_auto_save' not in st.session_state:
    st.session_state.last_auto_save = current_time

# Auto-save every 30 seconds if there's data
time_since_save = (current_time - st.session_state.last_auto_save).total_seconds()
if time_since_save > 30 and (st.session_state.people or st.session_state.firms or st.session_state.all_extractions):
    save_data()
    st.session_state.last_auto_save = current_time
