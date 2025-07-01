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

# --- MISSING FUNCTIONS - MOVED TO TOP ---

def handle_dynamic_input(field_name, current_value, table_name, context=""):
    """
    Handle dynamic input for fields with existing options and ability to add new ones
    
    Args:
        field_name: Name of the field (e.g., 'location', 'company')
        current_value: Current value to pre-select
        table_name: Database table name ('people', 'firms', etc.)
        context: Additional context for unique keys
    
    Returns:
        Selected or newly entered value
    """
    import streamlit as st
    
    # Get existing options from database based on field and table
    existing_options = get_unique_values_from_session_state(table_name, field_name)
    
    # Remove None/empty values and sort
    existing_options = sorted([opt for opt in existing_options if opt and opt.strip() and opt != 'Unknown'])
    
    # Add "Add New..." option
    options = [""] + existing_options + ["+ Add New"]
    
    # Find index of current value
    try:
        default_index = options.index(current_value) if current_value in options else 0
    except (ValueError, TypeError):
        default_index = 0
    
    # Create unique key for selectbox
    unique_key = f"{field_name}_select_{table_name}_{context}"
    
    # Create selectbox
    selected = st.selectbox(
        f"Select {field_name.replace('_', ' ').title()}",
        options=options,
        index=default_index,
        key=unique_key
    )
    
    # If "Add New" is selected, show text input
    if selected == "+ Add New":
        new_value_key = f"{field_name}_new_{table_name}_{context}"
        new_value = st.text_input(
            f"Enter new {field_name.replace('_', ' ')}:",
            placeholder=f"Enter {field_name.replace('_', ' ')}...",
            key=new_value_key
        )
        return new_value.strip() if new_value else ""
    
    return selected if selected else ""


def get_unique_values_from_session_state(table_name, field_name):
    """
    Get unique values from a specific field in session state data
    
    Args:
        table_name: Name of the data source ('people', 'firms', 'employments')
        field_name: Name of the field
    
    Returns:
        List of unique values
    """
    values = set()
    
    try:
        if table_name == "people":
            data_source = st.session_state.people
        elif table_name == "firms":
            data_source = st.session_state.firms
        elif table_name == "employments":
            data_source = st.session_state.employments
        else:
            return []
        
        for item in data_source:
            value = safe_get(item, field_name)
            if value and value.strip() and value != 'Unknown':
                values.add(value.strip())
        
        return list(values)
    
    except Exception as e:
        st.error(f"Error getting unique values for {field_name} from {table_name}: {str(e)}")
        return []


def global_search(query):
    """
    Global search function for people, firms, and performance metrics
    
    Args:
        query: Search query string
    
    Returns:
        Tuple of (matching_people, matching_firms, matching_metrics)
    """
    query_lower = query.lower().strip()
    
    if len(query_lower) < 2:
        return [], [], []
    
    matching_people = []
    matching_firms = []
    matching_metrics = []
    
    # Search people
    for person in st.session_state.people:
        searchable_text = " ".join([
            safe_get(person, 'name', ''),
            safe_get(person, 'current_title', ''),
            safe_get(person, 'current_company_name', ''),
            safe_get(person, 'location', ''),
            safe_get(person, 'expertise', ''),
            safe_get(person, 'strategy', ''),
            safe_get(person, 'education', '')
        ]).lower()
        
        if query_lower in searchable_text:
            matching_people.append(person)
    
    # Search firms
    for firm in st.session_state.firms:
        searchable_text = " ".join([
            safe_get(firm, 'name', ''),
            safe_get(firm, 'location', ''),
            safe_get(firm, 'strategy', ''),
            safe_get(firm, 'description', ''),
            safe_get(firm, 'headquarters', '')
        ]).lower()
        
        if query_lower in searchable_text:
            matching_firms.append(firm)
    
    # Search performance metrics in firms
    for firm in st.session_state.firms:
        if firm.get('performance_metrics'):
            for metric in firm['performance_metrics']:
                searchable_text = " ".join([
                    safe_get(metric, 'metric_type', ''),
                    safe_get(metric, 'period', ''),
                    safe_get(metric, 'additional_info', ''),
                    safe_get(firm, 'name', '')
                ]).lower()
                
                if query_lower in searchable_text:
                    matching_metrics.append({**metric, 'fund_name': firm['name']})
    
    return matching_people, matching_firms, matching_metrics

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
        
        # Save firms (now includes performance metrics)
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
    
    # Sample firms with detailed information and performance metrics
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
            "description": "Asia's largest hedge fund focusing on technology and healthcare investments",
            "performance_metrics": [
                {
                    "id": str(uuid.uuid4()),
                    "metric_type": "return",
                    "value": "12.5",
                    "period": "YTD",
                    "date": "2025",
                    "additional_info": "Net return"
                }
            ]
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
            "description": "Global hedge fund with significant Asian operations",
            "performance_metrics": [
                {
                    "id": str(uuid.uuid4()),
                    "metric_type": "sharpe",
                    "value": "1.8",
                    "period": "Current",
                    "date": "2025",
                    "additional_info": "Improved from 1.2"
                }
            ]
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
            "description": "Leading global hedge fund with growing Asian presence",
            "performance_metrics": []
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
        st.session_state.current_view = 'people'
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
    if 'global_search' not in st.session_state:
        st.session_state.global_search = ""
    
    # Pagination state
    if 'people_page' not in st.session_state:
        st.session_state.people_page = 0
    if 'firms_page' not in st.session_state:
        st.session_state.firms_page = 0
    if 'search_page' not in st.session_state:
        st.session_state.search_page = 0

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_cached_context():
    """Create cached context for hedge fund extraction to reduce token usage"""
    return {
        "system_instructions": """You are an expert hedge fund intelligence analyst specializing in extracting people movements and performance metrics from financial newsletters and industry reports.

EXTRACTION RULES:
1. Extract ALL people mentioned in professional contexts (hires, promotions, departures, launches)
2. Extract ALL performance metrics (returns, IRR, Sharpe, drawdown, AUM, alpha, beta, volatility)
3. Be precise with numbers - extract exact values without interpretation
4. Include time periods and benchmarks when mentioned
5. Focus on hedge funds, asset managers, and investment professionals

PEOPLE CATEGORIES:
- New hires and departures
- Promotions and role changes  
- Fund launches and closures
- Leadership appointments

PERFORMANCE METRICS:
- Returns (YTD, annual, quarterly, inception-to-date)
- Risk metrics (Sharpe ratio, maximum drawdown, volatility)
- Alpha and beta coefficients
- Assets under management (AUM)
- Benchmark comparisons and outperformance""",
        
        "example_input": """Goldman Sachs veteran John Smith joins Citadel Asia as Managing Director in Hong Kong. 
Former JPMorgan portfolio manager Lisa Chen launches her own hedge fund, Dragon Capital.
Engineers Gate topped $4 billion in assets and is up 12% this year. The fund's Sharpe ratio improved to 1.8.
Millennium's flagship fund returned 15.2% net in Q2 with maximum drawdown of 2.1%.""",
        
        "example_output": """{
  "people": [
    {
      "name": "John Smith",
      "company": "Citadel Asia", 
      "title": "Managing Director",
      "movement_type": "hire",
      "location": "Hong Kong"
    },
    {
      "name": "Lisa Chen",
      "company": "Dragon Capital",
      "title": "Founder", 
      "movement_type": "launch",
      "location": "Unknown"
    }
  ],
  "performance": [
    {
      "fund_name": "Engineers Gate",
      "metric_type": "aum",
      "value": "4000000000",
      "period": "Current",
      "date": "2025",
      "additional_info": "USD"
    },
    {
      "fund_name": "Engineers Gate", 
      "metric_type": "return",
      "value": "12",
      "period": "YTD",
      "date": "2025",
      "additional_info": "percent"
    },
    {
      "fund_name": "Engineers Gate",
      "metric_type": "sharpe",
      "value": "1.8", 
      "period": "Current",
      "date": "2025",
      "additional_info": "improved from 1.2"
    },
    {
      "fund_name": "Millennium",
      "metric_type": "return",
      "value": "15.2",
      "period": "Q2",
      "date": "2025",
      "additional_info": "net return, flagship fund"
    },
    {
      "fund_name": "Millennium",
      "metric_type": "drawdown", 
      "value": "2.1",
      "period": "Q2",
      "date": "2025",
      "additional_info": "maximum drawdown, percent"
    }
  ]
}""",
        
        "output_format": """{
  "people": [
    {
      "name": "Full Name",
      "company": "Company Name", 
      "title": "Job Title",
      "movement_type": "hire|promotion|launch|departure",
      "location": "City/Country"
    }
  ],
  "performance": [
    {
      "fund_name": "Fund Name",
      "metric_type": "return|irr|sharpe|drawdown|alpha|beta|volatility|aum",
      "value": "numeric_value_only",
      "period": "YTD|Q1|Q2|Q3|Q4|1Y|3Y|5Y|ITD|Current",
      "date": "YYYY or YYYY-MM-DD",
      "benchmark": "comparison_benchmark_if_mentioned", 
      "additional_info": "units_context_details"
    }
  ]
}"""
    }

def build_extraction_prompt_with_cache(newsletter_text, cached_context):
    """Build extraction prompt using cached context to minimize token usage"""
    
    prompt = f"""
{cached_context['system_instructions']}

CRITICAL EXTRACTION RULES:
1. NEVER use placeholder text like "Full Name" or "Company Name" 
2. If you cannot find a person's actual name, skip that entry entirely
3. Extract ONLY real, specific names and companies mentioned in the text
4. Be extra careful with abbreviated titles (CIO, PM, MD, etc.) - find the full context
5. Look for phrases like "appoints", "hires", "joins", "moves to", "promoted to"

EXAMPLE INPUT:
{cached_context['example_input']}

EXAMPLE OUTPUT:
{cached_context['example_output']}

REQUIRED OUTPUT FORMAT:
{cached_context['output_format']}

NOW EXTRACT FROM THIS NEWSLETTER:
{newsletter_text}

IMPORTANT: Only include entries where you have found ACTUAL names and companies. If you cannot find a specific person's name, do NOT create an entry with placeholder text. Return empty arrays if no specific information is found.

Return ONLY the JSON output with both people and performance arrays populated."""
    
    return prompt

def extract_single_chunk_safe(text, model):
    """Safe single chunk extraction with cached context and better validation"""
    try:
        # Use cached context to build efficient prompt
        cached_context = create_cached_context()
        prompt = build_extraction_prompt_with_cache(text, cached_context)
        
        response = model.generate_content(prompt)
        if not response or not response.text:
            return [], []
        
        # Show debug info if enabled
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            with st.expander("🐛 Debug: Raw AI Response", expanded=False):
                st.code(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
        
        # Parse JSON
        json_start = response.text.find('{')
        json_end = response.text.rfind('}') + 1
        
        if json_start == -1:
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.error("🐛 Debug: No JSON found in AI response")
            return [], []
        
        result = json.loads(response.text[json_start:json_end])
        people = result.get('people', [])
        performance = result.get('performance', [])
        
        # Enhanced validation to filter out placeholder text and incomplete extractions
        valid_people = []
        valid_performance = []
        
        for p in people:
            name = p.get('name', '').strip()
            company = p.get('company', '').strip()
            
            # Skip entries with placeholder text or incomplete data
            if (name and company and 
                name.lower() not in ['full name', 'name', 'person name', 'unknown'] and
                company.lower() not in ['company', 'company name', 'firm name', 'unknown'] and
                len(name) > 2 and len(company) > 2):
                valid_people.append(p)
        
        for p in performance:
            fund_name = p.get('fund_name', '').strip()
            metric_type = p.get('metric_type', '').strip()
            value = p.get('value', '').strip()
            
            # Skip entries with placeholder text or incomplete data
            if (fund_name and metric_type and value and
                fund_name.lower() not in ['fund name', 'fund', 'unknown'] and
                metric_type.lower() not in ['metric', 'metric type', 'unknown'] and
                value.lower() not in ['value', 'unknown', 'n/a']):
                valid_performance.append(p)
        
        return valid_people, valid_performance
        
    except Exception as e:
        st.warning(f"Single chunk failed: {str(e)[:100]}")
        return [], []

def extract_multi_chunk_safe(text, model, chunk_size=15000):
    """Enhanced multi-chunk processing with intelligent rate limiting"""
    
    try:
        # Smart chunking - try to break at paragraph boundaries
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + chunk_size, len(text))
            
            # If not at end of text, try to break at paragraph boundary
            if end_pos < len(text):
                # Look for paragraph break within last 500 chars
                search_start = max(end_pos - 500, current_pos)
                para_break = text.rfind('\n\n', search_start, end_pos)
                
                if para_break > current_pos:
                    end_pos = para_break + 2
            
            chunk = text[current_pos:end_pos].strip()
            if len(chunk) > 100:  # Skip tiny chunks
                chunks.append(chunk)
            
            current_pos = end_pos
        
        st.info(f"📄 Split into {len(chunks)} chunks for processing")
        
        # Intelligent delay based on model
        model_id = getattr(model, 'model_id', 'gemini-1.5-flash')
        if '1.5-pro' in model_id:
            delay = 35  # Conservative for Pro
        else:
            delay = 8   # Faster for Flash models
        
        # Process chunks with progress tracking
        all_people = []
        all_performance = []
        successful = 0
        failed = 0
        
        # Create progress container
        progress_container = st.container()
        
        for i, chunk in enumerate(chunks):
            try:
                with progress_container:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.info(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                    with col2:
                        st.metric("✅ Success", successful)
                    with col3:
                        st.metric("❌ Failed", failed)
                
                if i > 0:
                    st.info(f"⏱️ Rate limit delay: {delay}s...")
                    time.sleep(delay)
                
                # Extract from chunk
                chunk_people, chunk_performance = extract_single_chunk_safe(chunk, model)
                
                if chunk_people or chunk_performance:
                    # Simple deduplication
                    for person in chunk_people:
                        name_company = f"{person.get('name', '').lower()}|{person.get('company', '').lower()}"
                        if not any(f"{existing.get('name', '').lower()}|{existing.get('company', '').lower()}" == name_company 
                                 for existing in all_people):
                            all_people.append(person)
                    
                    for perf in chunk_performance:
                        fund_metric = f"{perf.get('fund_name', '').lower()}|{perf.get('metric_type', '').lower()}|{perf.get('period', '').lower()}"
                        if not any(f"{existing.get('fund_name', '').lower()}|{existing.get('metric_type', '').lower()}|{existing.get('period', '').lower()}" == fund_metric
                                 for existing in all_performance):
                            all_performance.append(perf)
                    
                    successful += 1
                    st.success(f"✅ Chunk {i+1}: Found {len(chunk_people)} people, {len(chunk_performance)} metrics")
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
                
                continue
        
        # Clear progress display
        progress_container.empty()
        
        # Final summary
        st.info(f"📊 **Processing Complete**: {successful} successful, {failed} failed chunks")
        st.success(f"🎯 **Total Extracted**: {len(all_people)} people, {len(all_performance)} performance metrics")
        
        return all_people, all_performance
        
    except Exception as e:
        st.error(f"Multi-chunk processing failed: {e}")
        return [], []

def extract_talent_simple(text, model):
    """Enhanced extraction with intelligent chunking - no file size limits"""
    if not model:
        return [], []
    
    # Preprocess and clean the text first
    cleaned_text = preprocess_newsletter_text(text)
    
    # More generous chunk size for better context
    max_single_chunk = 20000
    
    if len(cleaned_text) <= max_single_chunk:
        # Single chunk - simple and reliable
        st.info("📄 Processing as single chunk...")
        return extract_single_chunk_safe(cleaned_text, model)
    else:
        # Multi-chunk with intelligent processing
        st.info(f"📊 Large file detected ({len(cleaned_text):,} chars). Using intelligent chunking...")
        return extract_multi_chunk_safe(cleaned_text, model, max_single_chunk)

def preprocess_newsletter_text(text):
    """Clean and preprocess newsletter text to remove noise and focus on relevant content"""
    import re
    
    # Show original size
    original_size = len(text)
    
    # Step 1: Extract and preserve subject lines that contain relevant info
    subject_line = ""
    subject_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if subject_match:
        subject_line = subject_match.group(1).strip()
        # Keep subject if it contains hedge fund keywords
        hf_keywords_in_subject = ['appoints', 'joins', 'launches', 'hires', 'promotes', 'moves', 'cio', 'ceo', 'pm', 'portfolio manager', 'hedge fund', 'capital', 'management']
        if any(keyword in subject_line.lower() for keyword in hf_keywords_in_subject):
            text = f"NEWSLETTER SUBJECT: {subject_line}\n\n{text}"
    
    # Step 2: Remove email headers (but preserve subject if already extracted above)
    email_header_patterns = [
        r'From:\s*.*?\n',
        r'To:\s*.*?\n', 
        r'Sent:\s*.*?\n',
        r'Subject:\s*.*?\n',  # Remove original subject since we preserved it above
        r'Date:\s*.*?\n',
        r'Reply-To:\s*.*?\n',
        r'Return-Path:\s*.*?\n'
    ]
    
    for pattern in email_header_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Step 3: Remove URLs and tracking links
    url_patterns = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',  # Standard URLs
        r'<https?://[^>]+>',  # URLs in angle brackets
        r'urldefense\.proofpoint\.com[^\s]*',  # Proofpoint URLs
        r'pardot\.withintelligence\.com[^\s]*',  # Tracking URLs
        r'jpmorgan\.email\.streetcontxt\.net[^\s]*'  # Email tracking
    ]
    
    for pattern in url_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 4: Remove email disclaimers and legal text
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
    
    # Step 5: Remove HTML artifacts and email formatting
    html_patterns = [
        r'<[^>]+>',  # HTML tags
        r'&[a-zA-Z0-9#]+;',  # HTML entities
        r'\[cid:[^\]]+\]',  # Email embedded images
        r'________________________________+',  # Email separators
        r'\*\s*\|.*?\|\s*\*',  # Email table formatting
    ]
    
    for pattern in html_patterns:
        text = re.sub(pattern, '', text)
    
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
    
    # Performance-related keywords
    performance_keywords = [
        'irr', 'internal rate of return', 'sharpe', 'sharpe ratio', 'drawdown', 
        'maximum drawdown', 'max drawdown', 'alpha', 'beta', 'volatility', 'vol',
        'return', 'returns', 'performance', 'ytd', 'year to date', 'annualized',
        'net return', 'gross return', 'benchmark', 'outperformed', 'underperformed',
        'basis points', 'bps', '%', 'percent', 'up ', 'down ', 'gained', 'lost',
        'aum', 'assets', 'billion', 'million', 'fund size', 'nav', 'net asset value'
    ]
    
    # Additional keywords for better extraction
    movement_keywords = [
        'appoints', 'appointed', 'hiring', 'hired', 'departure', 'departing', 
        'leaving', 'joining', 'joined', 'moved', 'moving', 'promoted', 'promotion',
        'named', 'named as', 'becomes', 'became', 'takes over', 'steps down'
    ]
    
    # Combine all keywords
    all_keywords = hf_keywords + performance_keywords + movement_keywords
    
    # Split into paragraphs and keep relevant ones
    paragraphs = text.split('\n\n')
    relevant_paragraphs = []
    
    for para in paragraphs:
        para_lower = para.lower()
        if any(keyword in para_lower for keyword in all_keywords):
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

def map_performance_to_firms(performance_metrics):
    """Map performance metrics to firms where possible"""
    for metric in performance_metrics:
        fund_name = safe_get(metric, 'fund_name').lower()
        
        # Try to find matching firm
        matching_firm = None
        for firm in st.session_state.firms:
            firm_name = safe_get(firm, 'name').lower()
            if fund_name in firm_name or firm_name in fund_name:
                matching_firm = firm
                break
        
        if matching_firm:
            # Add metric to firm's performance_metrics
            if 'performance_metrics' not in matching_firm:
                matching_firm['performance_metrics'] = []
            
            # Add unique ID if not present
            if 'id' not in metric:
                metric['id'] = str(uuid.uuid4())
            
            # Check if metric already exists (avoid duplicates)
            existing = any(
                m.get('metric_type') == metric.get('metric_type') and 
                m.get('period') == metric.get('period') and
                m.get('date') == metric.get('date')
                for m in matching_firm['performance_metrics']
            )
            
            if not existing:
                matching_firm['performance_metrics'].append(metric)
    
    return performance_metrics

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

def get_person_performance_metrics(person_id):
    """Get performance metrics related to a specific person"""
    person = get_person_by_id(person_id)
    if not person:
        return []
    
    person_company = safe_get(person, 'current_company_name').lower()
    related_metrics = []
    
    # Look for metrics in firms
    for firm in st.session_state.firms:
        firm_name = safe_get(firm, 'name').lower()
        if person_company in firm_name or firm_name in person_company:
            if firm.get('performance_metrics'):
                related_metrics.extend(firm['performance_metrics'])
    
    return related_metrics

def get_firm_performance_metrics(firm_id):
    """Get performance metrics related to a specific firm"""
    firm = get_firm_by_id(firm_id)
    if not firm:
        return []
    
    return firm.get('performance_metrics', [])

# --- Pagination Helpers ---
def paginate_data(data, page, items_per_page=10):
    """Paginate data and return current page items and pagination info"""
    total_items = len(data)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    # Ensure page is within bounds
    page = max(0, min(page, total_pages - 1))
    
    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    current_items = data[start_idx:end_idx]
    
    return current_items, {
        'current_page': page,
        'total_pages': total_pages,
        'total_items': total_items,
        'items_per_page': items_per_page,
        'start_idx': start_idx,
        'end_idx': end_idx
    }

def display_pagination_controls(page_info, page_key):
    """Display pagination controls"""
    if page_info['total_pages'] <= 1:
        return
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=page_info['current_page'] == 0, key=f"{page_key}_first"):
            st.session_state[f"{page_key}_page"] = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Prev", disabled=page_info['current_page'] == 0, key=f"{page_key}_prev"):
            st.session_state[f"{page_key}_page"] = max(0, page_info['current_page'] - 1)
            st.rerun()
    
    with col3:
        st.write(f"Page {page_info['current_page'] + 1} of {page_info['total_pages']} " +
                f"(showing {page_info['start_idx'] + 1}-{page_info['end_idx']} of {page_info['total_items']})")
    
    with col4:
        if st.button("▶️ Next", disabled=page_info['current_page'] >= page_info['total_pages'] - 1, key=f"{page_key}_next"):
            st.session_state[f"{page_key}_page"] = min(page_info['total_pages'] - 1, page_info['current_page'] + 1)
            st.rerun()
    
    with col5:
        if st.button("⏭️ Last", disabled=page_info['current_page'] >= page_info['total_pages'] - 1, key=f"{page_key}_last"):
            st.session_state[f"{page_key}_page"] = page_info['total_pages'] - 1
            st.rerun()

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

def go_to_performance():
    st.session_state.current_view = 'performance'

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
    
    # Model Selection - FIXED MODEL OPTIONS
    st.markdown("---")
    st.subheader("🤖 Model Selection")
    
    model_options = {
        "Gemini 1.5 Flash (Recommended)": "gemini-1.5-flash",
        "Gemini 1.5 Pro (Advanced)": "gemini-1.5-pro", 
        "Gemini 2.0 Flash": "gemini-2.0-flash-exp",
        "Gemini 1.5 Flash Latest": "gemini-1.5-flash-latest"  # Fixed: removed invalid model
    }
    
    selected_model_name = st.selectbox(
        "Choose AI model:",
        options=list(model_options.keys()),
        index=0,  # Default to 1.5 Flash
        help="Different models have different capabilities and rate limits"
    )
    
    selected_model_id = model_options[selected_model_name]
    
    # Setup model with selected version
    model = None
    if api_key and GENAI_AVAILABLE:
        model = setup_gemini(api_key, selected_model_id)
        
        st.markdown("---")
        st.subheader("📄 Extract from Newsletter")
        
        # Input method - ENHANCED file handling with NO SIZE LIMITS
        input_method = st.radio("Input method:", ["📝 Text", "📁 File"])
        
        newsletter_text = ""
        if input_method == "📝 Text":
            newsletter_text = st.text_area("Newsletter content:", height=200, 
                                         placeholder="Paste hedge fund newsletter content here...")
        else:
            uploaded_file = st.file_uploader("Upload newsletter:", 
                                            type=['txt', 'doc', 'docx', 'pdf'], 
                                            help="✅ No size limits! Large files will be intelligently chunked.")
            if uploaded_file:
                try:
                    # Get file details
                    file_size = len(uploaded_file.getvalue())
                    file_size_mb = file_size / (1024 * 1024)
                    
                    st.info(f"📁 **File uploaded**: {uploaded_file.name} ({file_size_mb:.1f} MB)")
                    
                    # Handle different file types
                    if uploaded_file.type == "text/plain":
                        # Simple text file
                        raw_data = uploaded_file.getvalue()
                        try:
                            newsletter_text = raw_data.decode('utf-8')
                        except:
                            try:
                                newsletter_text = raw_data.decode('latin-1')
                            except:
                                st.error("Could not read file. Try saving as UTF-8 text file.")
                    else:
                        st.warning("Currently only .txt files are supported. Please convert your file to .txt format.")
                    
                    if newsletter_text:
                        char_count = len(newsletter_text)
                        estimated_chunks = max(1, char_count // 20000)
                        estimated_time = estimated_chunks * 2  # 2 minutes per chunk estimate
                        
                        st.success(f"✅ **File processed successfully!**")
                        st.info(f"• **Size**: {char_count:,} characters")
                        st.info(f"• **Estimated chunks**: {estimated_chunks}")
                        st.info(f"• **Estimated time**: ~{estimated_time} minutes")
                        
                        if estimated_chunks > 10:
                            st.warning("⚠️ **Large file detected** - Processing will take time but there are no size limits!")
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Debug mode toggle
        debug_mode = st.checkbox("🐛 Enable Debug Mode", help="Shows AI raw output before filtering")
        st.session_state.debug_mode = debug_mode

        # Extract button - ENHANCED with no size limits
        if st.button("🚀 Extract Talent", use_container_width=True):
            if not newsletter_text.strip():
                st.error("Please provide newsletter content")
            elif not model:
                st.error("Please provide API key")
            else:
                char_count = len(newsletter_text)
                st.info(f"📊 **Processing {char_count:,} characters** (No size limits!)")
                
                # Enhanced processing logic
                try:
                    st.info("🤖 **Starting extraction with intelligent chunking...**")
                    
                    # Show processing status
                    with st.status("Processing newsletter...", expanded=True) as status:
                        st.write("🔄 Initializing extraction...")
                        people_extractions, performance_extractions = extract_talent_simple(newsletter_text, model)
                        
                        if people_extractions or performance_extractions:
                            st.write(f"✅ Found {len(people_extractions)} people, {len(performance_extractions)} performance metrics!")
                            status.update(label="✅ Extraction complete!", state="complete")
                        else:
                            st.write("⚠️ No data found")
                            status.update(label="⚠️ No results found", state="complete")
                    
                    if people_extractions or performance_extractions:
                        # Add metadata to extractions
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for ext in people_extractions:
                            ext['timestamp'] = timestamp
                            ext['id'] = str(uuid.uuid4())
                        
                        # Map performance metrics to firms
                        if performance_extractions:
                            performance_extractions = map_performance_to_firms(performance_extractions)
                        
                        # Process people extractions and detect updates
                        if people_extractions:
                            st.info("🔍 Checking for existing people and potential updates...")
                            new_people, pending_updates = process_extractions_with_update_detection(people_extractions)
                            
                            # Add pending updates to session state
                            if pending_updates:
                                st.session_state.pending_updates.extend(pending_updates)
                                st.session_state.show_update_review = True
                        else:
                            new_people = []
                            pending_updates = []
                        
                        # Save new extractions
                        st.session_state.all_extractions.extend(people_extractions)
                        
                        # Save results
                        if save_data():
                            st.success(f"🎉 **Extraction Complete!**")
                        else:
                            st.error("⚠️ Extraction successful but save failed!")
                        
                        # Show comprehensive summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("👥 New People", len(new_people))
                        with col2:
                            st.metric("🔄 Pending Updates", len(pending_updates))
                        with col3:
                            st.metric("📊 Performance Metrics", len(performance_extractions))
                        with col4:
                            st.metric("🏢 Total Extracted", len(people_extractions))
                        
                        # Show what happened
                        if new_people:
                            with st.expander(f"📋 {len(new_people)} New People Found"):
                                for person in new_people[:10]:  # Show first 10
                                    st.write(f"• **{person['name']}** → {person['company']}")
                                if len(new_people) > 10:
                                    st.write(f"... and {len(new_people) - 10} more")
                        
                        if performance_extractions:
                            with st.expander(f"📊 {len(performance_extractions)} Performance Metrics Found"):
                                for perf in performance_extractions[:10]:  # Show first 10
                                    metric_display = f"{perf.get('fund_name', 'Unknown')} - {perf.get('metric_type', 'Unknown')}: {perf.get('value', 'N/A')}"
                                    if perf.get('period'):
                                        metric_display += f" ({perf.get('period')})"
                                    st.write(f"• {metric_display}")
                                
                                if len(performance_extractions) > 10:
                                    st.write(f"... and {len(performance_extractions) - 10} more")
                        
                        if pending_updates:
                            st.warning(f"⚠️ Found {len(pending_updates)} potential updates for existing people!")
                            st.info("👆 **Review updates in the sidebar before they're applied**")
                        
                    else:
                        st.warning("⚠️ No people or performance data found. Try a different model or check content.")
                        
                except Exception as e:
                    st.error(f"💥 **Extraction failed**: {str(e)}")
                    st.info("**Try**: Different model or copy/paste instead of file upload")
        
        # Show recent extractions
        if st.session_state.all_extractions:
            st.markdown("---")
            st.subheader("📊 Recent Extractions")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👥 People", len(st.session_state.all_extractions))
            with col2:
                # Count total performance metrics across all firms
                total_metrics = sum(len(firm.get('performance_metrics', [])) for firm in st.session_state.firms)
                st.metric("📊 Performance", total_metrics)
            
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
                                "description": f"Hedge fund - extracted from newsletter",
                                "performance_metrics": []
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

    elif not GENAI_AVAILABLE:
        st.error("Please install: pip install google-generativeai")

# --- MAIN CONTENT AREA ---
st.title("👥 Asian Hedge Fund Talent Network")
st.markdown("### Professional intelligence platform for Asia's hedge fund industry")

# --- GLOBAL SEARCH BAR ---
st.markdown("---")
col1, col2 = st.columns([4, 1])

with col1:
    search_query = st.text_input(
        "🔍 Search people, firms, or performance...", 
        value=st.session_state.global_search,
        placeholder="Try: 'Goldman Sachs', 'Portfolio Manager', 'Citadel', 'Sharpe ratio'...",
        key="global_search_input"
    )

with col2:
    if st.button("🔍 Search", use_container_width=True):
        st.session_state.global_search = search_query
        st.rerun()

# Handle global search results with pagination
if search_query and len(search_query.strip()) >= 2:
    st.session_state.global_search = search_query
    matching_people, matching_firms, matching_metrics = global_search(search_query)
    
    if matching_people or matching_firms or matching_metrics:
        st.markdown("### 🔍 Search Results")
        
        # Show search results with pagination
        if matching_people:
            st.markdown(f"**👥 People ({len(matching_people)} found)**")
            people_to_show, people_page_info = paginate_data(matching_people, st.session_state.search_page, 5)
            
            for person in people_to_show:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{safe_get(person, 'name')}**")
                    st.caption(f"{safe_get(person, 'current_title')} at {safe_get(person, 'current_company_name')}")
                with col2:
                    st.caption(f"📍 {safe_get(person, 'location')}")
                with col3:
                    if st.button("👁️ View", key=f"search_person_{person['id']}", use_container_width=True):
                        go_to_person_details(person['id'])
                        st.rerun()
            
            if len(matching_people) > 5:
                display_pagination_controls(people_page_info, "search")
        
        # Clear search button
        if st.button("❌ Clear Search"):
            st.session_state.global_search = ""
            st.rerun()
        
        st.markdown("---")
    
    else:
        st.info(f"🔍 No results found for '{search_query}'. Try different keywords.")
        if st.button("❌ Clear Search"):
            st.session_state.global_search = ""
            st.rerun()
        st.markdown("---")

# Top Navigation
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 2])

with col1:
    if st.button("👥 People", use_container_width=True, 
                 type="primary" if st.session_state.current_view == 'people' else "secondary"):
        go_to_people()
        st.rerun()

with col2:
    if st.button("🏢 Firms", use_container_width=True, 
                 type="primary" if st.session_state.current_view == 'firms' else "secondary"):
        go_to_firms()
        st.rerun()

with col3:
    if st.button("📊 Performance", use_container_width=True, 
                 type="primary" if st.session_state.current_view == 'performance' else "secondary"):
        go_to_performance()
        st.rerun()

with col4:
    if st.button("➕ Add Person", use_container_width=True):
        st.session_state.show_add_person_modal = True
        st.rerun()

with col5:
    if st.button("🏢➕ Add Firm", use_container_width=True):
        st.session_state.show_add_firm_modal = True
        st.rerun()

with col6:
    # Quick stats
    col6a, col6b, col6c = st.columns(3)
    with col6a:
        st.metric("People", len(st.session_state.people))
    with col6b:
        st.metric("Firms", len(st.session_state.firms))
    with col6c:
        # Count total performance metrics across all firms
        total_metrics = sum(len(firm.get('performance_metrics', [])) for firm in st.session_state.firms)
        st.metric("Metrics", total_metrics)

# --- ADD PERSON MODAL ---
if st.session_state.show_add_person_modal:
    st.markdown("---")
    st.subheader("➕ Add New Person")
    
    with st.form("add_person_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", placeholder="John Smith")
            title = st.text_input("Current Title*", placeholder="Portfolio Manager")
            
            # Dynamic company selection
            company_options = [""] + [f['name'] for f in st.session_state.firms]
            company = st.selectbox("Current Company*", options=company_options)
            
            # Dynamic location selection
            location = handle_dynamic_input("location", "", "people", "add_person")
        
        with col2:
            email = st.text_input("Email", placeholder="john.smith@company.com")
            phone = st.text_input("Phone", placeholder="+852-1234-5678")
            education = st.text_input("Education", placeholder="Harvard, MIT")
            
            # Dynamic expertise selection
            expertise = handle_dynamic_input("expertise", "", "people", "add_person")
        
        # Additional fields
        col3, col4 = st.columns(2)
        with col3:
            aum_managed = st.text_input("AUM Managed", placeholder="500M USD")
        with col4:
            # Dynamic strategy selection
            strategy = handle_dynamic_input("strategy", "", "people", "add_person")
        
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
                    "aum_managed": aum_managed,
                    "strategy": strategy
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
                    "strategy": strategy or "Unknown"
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
            
            # Dynamic location selection
            location = handle_dynamic_input("location", "", "firms", "add_firm")
            
            aum = st.text_input("AUM", placeholder="5B USD")
            
        with col2:
            # Dynamic strategy selection
            strategy = handle_dynamic_input("strategy", "", "firms", "add_firm")
            
            founded = st.number_input("Founded", min_value=1900, max_value=2025, value=2000)
            website = st.text_input("Website", placeholder="https://company.com")
        
        description = st.text_area("Description", placeholder="Brief description of the firm...")
        
        submitted = st.form_submit_button("Add Firm")
        
        if submitted:
            if firm_name and location:
                st.session_state.firms.append({
                    "id": str(uuid.uuid4()),
                    "name": firm_name,
                    "location": location,
                    "headquarters": location,  # Default headquarters to location
                    "aum": aum,
                    "founded": founded if founded > 1900 else None,
                    "strategy": strategy,
                    "website": website,
                    "description": description if description else f"{strategy} hedge fund based in {location}",
                    "performance_metrics": []  # Initialize empty performance metrics
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

# --- PEOPLE VIEW WITH PAGINATION ---
if st.session_state.current_view == 'people':
    st.markdown("---")
    st.header("👥 Hedge Fund Professionals")
    
    if not st.session_state.people:
        st.info("No people added yet. Use 'Add Person' button above or extract from newsletters using AI.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            # Dynamic location filter from existing data
            locations = ["All"] + sorted(list(set(safe_get(p, 'location') for p in st.session_state.people if safe_get(p, 'location') not in ['Unknown', 'N/A', ''])))
            location_filter = st.selectbox("Filter by Location", locations)
        with col2:
            # Dynamic company filter from existing data
            companies = ["All"] + sorted(list(set(safe_get(p, 'current_company_name') for p in st.session_state.people if safe_get(p, 'current_company_name') not in ['Unknown', 'N/A', ''])))
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
        
        # Paginate results
        people_to_show, people_page_info = paginate_data(filtered_people, st.session_state.people_page, 10)
        
        st.write(f"**Showing {people_page_info['start_idx'] + 1}-{people_page_info['end_idx']} of {people_page_info['total_items']} people**")
        
        # Display people in compact cards
        for person in people_to_show:
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
                        # Show performance metrics count if available
                        person_metrics = get_person_performance_metrics(person['id'])
                        if person_metrics:
                            st.metric("📊", f"{len(person_metrics)} metrics", label_visibility="collapsed")
                        else:
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
        
        # Pagination controls
        display_pagination_controls(people_page_info, "people")

# --- FIRMS VIEW WITH PAGINATION ---
elif st.session_state.current_view == 'firms':
    st.markdown("---")
    st.header("🏢 Hedge Funds in Asia")
    
    if not st.session_state.firms:
        st.info("No firms added yet. Use 'Add Firm' button above.")
    else:
        # Paginate results
        firms_to_show, firms_page_info = paginate_data(st.session_state.firms, st.session_state.firms_page, 10)
        
        st.write(f"**Showing {firms_page_info['start_idx'] + 1}-{firms_page_info['end_idx']} of {firms_page_info['total_items']} firms**")
        
        # Display firms in compact cards
        for firm in firms_to_show:
            people_count = len(get_people_by_firm(safe_get(firm, 'name')))
            metrics_count = len(firm.get('performance_metrics', []))
            
            # Compact card design
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**🏢 {safe_get(firm, 'name')}**")
                    st.caption(f"{safe_get(firm, 'strategy')} • {safe_get(firm, 'location')}")
                
                with col2:
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        st.metric("💰", safe_get(firm, 'aum'), label_visibility="collapsed")
                    with col2b:
                        st.metric("👥", people_count, label_visibility="collapsed")
                    with col2c:
                        st.metric("📊", metrics_count, label_visibility="collapsed")
                
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
        
        # Pagination controls
        display_pagination_controls(firms_page_info, "firms")

# --- PERFORMANCE METRICS VIEW (Updated to use firm-linked metrics) ---
elif st.session_state.current_view == 'performance':
    st.markdown("---")
    st.header("📊 Hedge Fund Performance Metrics")
    
    # Collect all performance metrics from firms
    all_performance_metrics = []
    for firm in st.session_state.firms:
        if firm.get('performance_metrics'):
            for metric in firm['performance_metrics']:
                # Add firm name to metric for display
                metric_with_firm = {**metric, 'fund_name': firm['name']}
                all_performance_metrics.append(metric_with_firm)
    
    if not all_performance_metrics:
        st.info("No performance metrics extracted yet. Use the AI extractor in the sidebar to analyze newsletters.")
    else:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            # Get unique fund names
            fund_names = ["All"] + sorted(list(set(safe_get(p, 'fund_name') for p in all_performance_metrics if safe_get(p, 'fund_name') != 'Unknown')))
            fund_filter = st.selectbox("Filter by Fund", fund_names)
        with col2:
            # Get unique metric types
            metric_types = ["All"] + sorted(list(set(safe_get(p, 'metric_type') for p in all_performance_metrics if safe_get(p, 'metric_type') != 'Unknown')))
            metric_filter = st.selectbox("Filter by Metric Type", metric_types)
        with col3:
            # Get unique periods
            periods = ["All"] + sorted(list(set(safe_get(p, 'period') for p in all_performance_metrics if safe_get(p, 'period') != 'Unknown')))
            period_filter = st.selectbox("Filter by Period", periods)
        
        # Apply filters
        filtered_metrics = all_performance_metrics
        if fund_filter != "All":
            filtered_metrics = [p for p in filtered_metrics if safe_get(p, 'fund_name') == fund_filter]
        if metric_filter != "All":
            filtered_metrics = [p for p in filtered_metrics if safe_get(p, 'metric_type') == metric_filter]
        if period_filter != "All":
            filtered_metrics = [p for p in filtered_metrics if safe_get(p, 'period') == period_filter]
        
        st.write(f"**Showing {len(filtered_metrics)} performance metrics**")
        
        if filtered_metrics:
            # Create DataFrame for display
            metrics_data = []
            for metric in filtered_metrics:
                metrics_data.append({
                    "Fund": safe_get(metric, 'fund_name'),
                    "Metric": safe_get(metric, 'metric_type'),
                    "Value": safe_get(metric, 'value'),
                    "Period": safe_get(metric, 'period'),
                    "Date": safe_get(metric, 'date'),
                    "Additional Info": safe_get(metric, 'additional_info')
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Display as interactive table
            st.dataframe(df_metrics, use_container_width=True)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                unique_funds = len(set(safe_get(m, 'fund_name') for m in filtered_metrics))
                st.metric("Unique Funds", unique_funds)
            with col2:
                unique_metrics = len(set(safe_get(m, 'metric_type') for m in filtered_metrics))
                st.metric("Metric Types", unique_metrics)
            with col3:
                latest_date = max([safe_get(m, 'date', '1900-01-01') for m in filtered_metrics])
                st.metric("Latest Data", latest_date)
            with col4:
                total_metrics = len(filtered_metrics)
                st.metric("Total Metrics", total_metrics)

# --- Footer ---
st.markdown("---")
st.markdown("### 👥 Asian Hedge Fund Talent Intelligence Platform")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**🔍 Global Search**")
with col2:
    st.markdown("**📊 Performance Tracking**") 
with col3:
    st.markdown("**🤝 Professional Networks**")
with col4:
    st.markdown("**🚀 Intelligent Chunking**")

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
            
            # Dynamic company selection
            current_company = safe_get(person_data, 'current_company_name')
            company_options = [""] + [f['name'] for f in st.session_state.firms]
            company_index = 0
            if current_company and current_company in company_options:
                company_index = company_options.index(current_company)
            company = st.selectbox("Current Company*", options=company_options, index=company_index)
            
            # Dynamic location selection
            location = handle_dynamic_input("location", safe_get(person_data, 'location'), "people", "edit_person")
        
        with col2:
            email = st.text_input("Email", value=safe_get(person_data, 'email'))
            phone = st.text_input("Phone", value=safe_get(person_data, 'phone'))
            linkedin = st.text_input("LinkedIn URL", value=safe_get(person_data, 'linkedin_profile_url'))
            education = st.text_input("Education", value=safe_get(person_data, 'education'))
        
        col3, col4 = st.columns(2)
        with col3:
            # Dynamic expertise selection
            expertise = handle_dynamic_input("expertise", safe_get(person_data, 'expertise'), "people", "edit_person")
            aum = st.text_input("AUM Managed", value=safe_get(person_data, 'aum_managed'))
        
        with col4:
            # Dynamic strategy selection
            strategy = handle_dynamic_input("strategy", safe_get(person_data, 'strategy'), "people", "edit_person")
        
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
            
            # Dynamic location selection
            location = handle_dynamic_input("location", safe_get(firm_data, 'location'), "firms", "edit_firm")
            
            headquarters = st.text_input("Headquarters", value=safe_get(firm_data, 'headquarters'))
            aum = st.text_input("AUM", value=safe_get(firm_data, 'aum'))
            
        with col2:
            # Dynamic strategy selection
            strategy = handle_dynamic_input("strategy", safe_get(firm_data, 'strategy'), "firms", "edit_firm")
            
            founded = st.number_input("Founded", min_value=1900, max_value=2025, 
                                    value=firm_data.get('founded', 2000) if firm_data.get('founded') else 2000)
            website = st.text_input("Website", value=safe_get(firm_data, 'website'))
        
        description = st.text_area("Description", value=safe_get(firm_data, 'description'))
        
        # Performance Metrics Management
        st.markdown("---")
        st.subheader("📊 Performance Metrics")
        
        existing_metrics = firm_data.get('performance_metrics', [])
        if existing_metrics:
            st.write(f"**Current Metrics ({len(existing_metrics)}):**")
            for i, metric in enumerate(existing_metrics):
                with st.expander(f"{metric.get('metric_type', 'Unknown')} - {metric.get('period', 'Unknown')}"):
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.write(f"**Value**: {metric.get('value', 'N/A')}")
                        st.write(f"**Period**: {metric.get('period', 'N/A')}")
                    with col_metric2:
                        st.write(f"**Date**: {metric.get('date', 'N/A')}")
                        st.write(f"**Info**: {metric.get('additional_info', 'N/A')}")
                    
                    if st.button(f"🗑️ Remove Metric", key=f"remove_metric_{i}"):
                        existing_metrics.pop(i)
                        firm_data['performance_metrics'] = existing_metrics
                        st.rerun()
        else:
            st.info("No performance metrics yet. Metrics will be added automatically when extracted from newsletters.")
        
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
    
    # Performance Metrics
    st.markdown("---")
    st.subheader("📊 Performance Metrics")
    
    firm_metrics = firm.get('performance_metrics', [])
    if firm_metrics:
        st.write(f"**Found {len(firm_metrics)} performance metrics:**")
        
        for metric in firm_metrics:
            metric_type = safe_get(metric, 'metric_type')
            value = safe_get(metric, 'value')
            period = safe_get(metric, 'period')
            date = safe_get(metric, 'date')
            additional_info = safe_get(metric, 'additional_info')
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{metric_type.title()}**")
            with col2:
                st.code(f"{value}%") if metric_type in ['return', 'sharpe', 'alpha'] else st.code(value)
            with col3:
                period_info = f"{period}" if period != 'Unknown' else ""
                date_info = f"({date})" if date != 'Unknown' else ""
                st.caption(f"{period_info} {date_info}")
            
            if additional_info and additional_info != 'Unknown':
                st.caption(f"ℹ️ {additional_info}")
    else:
        st.info("No performance metrics found for this firm.")
        st.write("💡 Performance metrics will appear here when extracted from newsletters.")
    
    # People at this firm
    st.markdown("---")
    st.subheader(f"👥 People at {safe_get(firm, 'name')}")
    
    firm_people = get_people_by_firm(safe_get(firm, 'name'))
    if firm_people:
        # Paginate people if there are many
        people_to_show, people_page_info = paginate_data(firm_people, 0, 10)
        
        for person in people_to_show:
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
        
        if len(firm_people) > 10:
            st.info(f"Showing first {len(people_to_show)} of {len(firm_people)} people")
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
    
    # Performance Metrics
    st.markdown("---")
    st.subheader("📊 Performance Track Record")
    
    person_metrics = get_person_performance_metrics(person['id'])
    if person_metrics:
        st.write(f"**Found {len(person_metrics)} performance metrics:**")
        
        for metric in person_metrics:
            metric_type = safe_get(metric, 'metric_type')
            value = safe_get(metric, 'value')
            period = safe_get(metric, 'period')
            date = safe_get(metric, 'date')
            additional_info = safe_get(metric, 'additional_info')
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{metric_type.title()}**")
            with col2:
                st.code(f"{value}%") if metric_type in ['return', 'sharpe', 'alpha'] else st.code(value)
            with col3:
                period_info = f"{period}" if period != 'Unknown' else ""
                date_info = f"({date})" if date != 'Unknown' else ""
                st.caption(f"{period_info} {date_info}")
            
            if additional_info and additional_info != 'Unknown':
                st.caption(f"ℹ️ {additional_info}")
    
    else:
        st.info("No performance metrics found for this person.")
        st.write("💡 Performance metrics will appear here when extracted from newsletters.")
    
    # Shared Work History
    st.markdown("---")
    st.subheader("🤝 Professional Network Connections")
    
    shared_history = get_shared_work_history(person['id'])
    
    if shared_history:
        st.write(f"**Found {len(shared_history)} colleagues who worked at the same companies:**")
        
        # Show top connections with pagination
        connections_to_show, connections_page_info = paginate_data(shared_history, 0, 10)
        
        for connection in connections_to_show:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{connection['colleague_name']}**")
                st.caption(f"Shared: {connection['shared_company']}")
            with col2:
                st.write(f"{connection['colleague_current_title']}")
                st.caption(f"at {connection['colleague_current_company']}")
            with col3:
                st.metric("Years Together", f"{connection['overlap_years']}")
                if st.button("👁️", key=f"view_colleague_{connection['colleague_id']}", help="View Profile"):
                    go_to_person_details(connection['colleague_id'])
                    st.rerun()
        
        if len(shared_history) > 10:
            st.info(f"Showing top {len(connections_to_show)} of {len(shared_history)} connections")
        
    else:
        st.info("No shared work history found with other people in the database.")
        st.write("💡 Add more people who worked at the same companies to see connections!")

# Force save data periodically
current_time = datetime.now()
if 'last_auto_save' not in st.session_state:
    st.session_state.last_auto_save = current_time

# Auto-save every 30 seconds if there's data
time_since_save = (current_time - st.session_state.last_auto_save).total_seconds()
if time_since_save > 30 and (st.session_state.people or st.session_state.firms or st.session_state.all_extractions):
    save_data()
    st.session_state.last_auto_save = current_time
