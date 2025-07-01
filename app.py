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
import threading # ADDED: Import threading for asynchronous saves

# Additional imports for enhanced export functionality
import zipfile
from io import BytesIO, StringIO

# Try to import openpyxl for Excel exports
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    st.sidebar.warning("📊 Excel export unavailable. Install openpyxl: pip install openpyxl")

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
    Enhanced dynamic input that prioritizes typing with suggestions
    
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
    
    # Create unique key for input
    unique_key = f"{field_name}_input_{table_name}_{context}"
    
    # Primary text input with current value
    user_input = st.text_input(
        f"{field_name.replace('_', ' ').title()}",
        value=current_value if current_value and current_value != 'Unknown' else "",
        placeholder=f"Enter {field_name.replace('_', ' ')} or select from suggestions below",
        key=unique_key,
        help=f"Type directly or choose from {len(existing_options)} existing options below"
    )
    
    # Show existing options as clickable suggestions if there are any
    if existing_options and len(existing_options) > 0:
        st.caption(f"💡 **Suggestions** (click to use):")
        
        # Display suggestions in columns for better layout
        cols_per_row = 3
        suggestion_cols = st.columns(cols_per_row)
        
        for i, option in enumerate(existing_options[:9]):  # Show max 9 suggestions
            col_idx = i % cols_per_row
            with suggestion_cols[col_idx]:
                # Use a button that updates the input when clicked
                if st.button(f"📍 {option}", key=f"{unique_key}_suggestion_{i}", help=f"Use: {option}"):
                    # Return the selected suggestion
                    st.session_state[unique_key] = option
                    st.rerun()
        
        if len(existing_options) > 9:
            st.caption(f"... and {len(existing_options) - 9} more options available")
    
    # Return the user input (either typed or from session state if suggestion was clicked)
    return user_input.strip() if user_input else ""

def enhanced_global_search(query):
    """
    Enhanced global search function with better matching and debugging
    """
    query_lower = query.lower().strip()
    
    if len(query_lower) < 2:
        return [], [], []
    
    matching_people = []
    matching_firms = []
    matching_metrics = []
    
    # Search people with enhanced matching
    for person in st.session_state.people:
        # Create comprehensive searchable text
        searchable_fields = [
            safe_get(person, 'name', ''),
            safe_get(person, 'current_title', ''),
            safe_get(person, 'current_company_name', ''),
            safe_get(person, 'location', ''),
            safe_get(person, 'expertise', ''),
            safe_get(person, 'strategy', ''),
            safe_get(person, 'education', ''),
            safe_get(person, 'email', ''),
            safe_get(person, 'aum_managed', '')
        ]
        
        searchable_text = " ".join([field for field in searchable_fields if field and field != 'Unknown']).lower()
        
        # Multiple search methods
        if (query_lower in searchable_text or 
            any(query_lower in field.lower() for field in searchable_fields if field and field != 'Unknown')):
            matching_people.append(person)
    
    # Search firms with enhanced matching  
    for firm in st.session_state.firms:
        searchable_fields = [
            safe_get(firm, 'name', ''),
            safe_get(firm, 'location', ''),
            safe_get(firm, 'strategy', ''),
            safe_get(firm, 'description', ''),
            safe_get(firm, 'headquarters', ''),
            safe_get(firm, 'aum', ''),
            safe_get(firm, 'website', '')
        ]
        
        searchable_text = " ".join([field for field in searchable_fields if field and field != 'Unknown']).lower()
        
        if (query_lower in searchable_text or 
            any(query_lower in field.lower() for field in searchable_fields if field and field != 'Unknown')):
            matching_firms.append(firm)
    
    # Search performance metrics in firms
    for firm in st.session_state.firms:
        if firm.get('performance_metrics'):
            for metric in firm['performance_metrics']:
                searchable_fields = [
                    safe_get(metric, 'metric_type', ''),
                    safe_get(metric, 'period', ''),
                    safe_get(metric, 'additional_info', ''),
                    safe_get(metric, 'value', ''),
                    safe_get(firm, 'name', '')
                ]
                
                searchable_text = " ".join([field for field in searchable_fields if field and field != 'Unknown']).lower()
                
                if (query_lower in searchable_text or 
                    any(query_lower in field.lower() for field in searchable_fields if field and field != 'Unknown')):
                    matching_metrics.append({**metric, 'fund_name': firm['name']})
    
    return matching_people, matching_firms, matching_metrics

# --- Database Persistence Setup ---
DATA_DIR = Path("hedge_fund_data")
DATA_DIR.mkdir(exist_ok=True)

PEOPLE_FILE = DATA_DIR / "people.json"
FIRMS_FILE = DATA_DIR / "firms.json"
EMPLOYMENTS_FILE = DATA_DIR / "employments.json"
EXTRACTIONS_FILE = DATA_DIR / "extractions.json"

# RENAMED: This function now performs the actual file saving in a thread-safe manner
def _perform_data_save_to_files(people_data, firms_data, employments_data, all_extractions_data):
    """
    Internal function to save data to JSON files.
    Designed to be called from a separate thread.
    """
    try:
        DATA_DIR.mkdir(exist_ok=True)
        
        with open(PEOPLE_FILE, 'w', encoding='utf-8') as f:
            json.dump(people_data, f, indent=2, default=str)
        
        with open(FIRMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(firms_data, f, indent=2, default=str)
        
        with open(EMPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(employments_data, f, indent=2, default=str)
        
        if all_extractions_data is not None:
            with open(EXTRACTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_extractions_data, f, indent=2, default=str)
        
        print(f"💾 Background save complete.") # Use print for console logging from thread
        # In a real app, you might use a queue or another mechanism to update UI with success
        if 'save_status' in st.session_state: # Example of updating status in main thread
            st.session_state.save_status = "Data saved successfully!"
        
    except Exception as e:
        print(f"❌ Background save error: {e}") # Use print for console logging from thread
        if 'save_status' in st.session_state:
            st.session_state.save_status = f"Error during save: {e}"


def save_data_async():
    """
    Triggers an asynchronous save of all extracted data to files.
    This function immediately returns without waiting for the save to complete.
    """
    # Create a copy of the data from session state before passing to thread
    # This is crucial as st.session_state is not thread-safe.
    people_copy = list(st.session_state.people)
    firms_copy = list(st.session_state.firms)
    employments_copy = list(st.session_state.employments)
    all_extractions_copy = list(st.session_state.all_extractions) if 'all_extractions' in st.session_state else None

    # Set a status message in the main thread (optional)
    st.session_state.save_status = "Saving data in background..."

    # Start the save operation in a new thread
    save_thread = threading.Thread(
        target=_perform_data_save_to_files,
        args=(people_copy, firms_copy, employments_copy, all_extractions_copy)
    )
    save_thread.daemon = True # Allow the program to exit even if thread is running
    save_thread.start()
    
    st.sidebar.info("💾 Auto-save triggered in background (non-blocking).") # Inform user

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
    
    # NEW: File processing preferences
    if 'preprocessing_mode' not in st.session_state:
        st.session_state.preprocessing_mode = "balanced"
    if 'chunk_size_preference' not in st.session_state:
        st.session_state.chunk_size_preference = "auto"
    
    # NEW: Review system
    if 'enable_review_mode' not in st.session_state:
        st.session_state.enable_review_mode = True
    if 'pending_review_data' not in st.session_state:
        st.session_state.pending_review_data = []
    if 'review_start_time' not in st.session_state:
        st.session_state.review_start_time = None
    if 'show_review_interface' not in st.session_state:
        st.session_state.show_review_interface = False
    if 'auto_save_timeout' not in st.session_state:
        st.session_state.auto_save_timeout = 180  # 3 minutes in seconds
    # ADDED: State for asynchronous save status
    if 'save_status' not in st.session_state:
        st.session_state.save_status = ""

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
    """Create cached context for hedge fund extraction with optimized prompts"""
    return {
        "system_instructions": """You are an expert financial analyst specializing in the hedge fund industry. Your task is to meticulously analyze the following text and extract key intelligence about hedge funds, investment banks, asset managers, private equity firms, and related financial institutions.

CORE EXTRACTION TARGETS:
1. PEOPLE: All individuals in professional contexts (current employees, new hires, departures, promotions, launches, appointments)
2. FIRMS: Hedge funds, investment banks, asset managers, family offices, private equity, sovereign wealth funds
3. PERFORMANCE DATA: Returns, risk metrics, AUM figures, fund performance, benchmarks
4. MOVEMENTS: Job changes, fund launches, firm transitions, strategic shifts

SPECIFIC FOCUS AREAS:
- Hedge fund managers and portfolio managers
- Investment bank professionals (VP, MD, Managing Director levels)
- Asset management executives (CIO, CEO, Head of Trading, etc.)
- Quantitative analysts and researchers
- Fund launches, closures, and strategic changes
- Performance attribution and risk metrics
- Assets under management (AUM) changes
- Geographic expansion and office openings

GEOGRAPHIC INTELLIGENCE:
- Identify primary geographic focus (Asia-Pacific, North America, Europe, etc.)
- Extract specific office locations and expansion plans
- Note regulatory environments and market access

PERFORMANCE METRICS PRIORITY:
- Net returns (YTD, annual, multi-year)
- Risk-adjusted returns (Sharpe ratio, information ratio)
- Maximum drawdown and volatility measures
- Alpha generation and beta coefficients
- Assets under management (AUM) and flows
- Benchmark comparisons and relative performance

FIRM CATEGORIZATION:
- Hedge funds (long/short equity, macro, credit, quantitative, etc.)
- Investment banks (bulge bracket, boutique, regional)
- Asset managers (traditional, alternative, specialized)
- Family offices (single-family, multi-family)
- Private equity and venture capital
- Sovereign wealth funds and pension funds""",
        
        "example_input": """Goldman Sachs veteran John Smith joins Citadel Asia as Managing Director in Hong Kong, bringing 15 years of equity trading experience. Former JPMorgan portfolio manager Lisa Chen launches Dragon Capital Management, a $200M long/short equity fund focused on Asian markets. 

Engineers Gate's systematic trading fund topped $4.2 billion in assets and delivered 12.3% net returns year-to-date, with a Sharpe ratio of 1.8 compared to 1.2 last year. The fund's maximum drawdown remained below 2.5% during Q3 volatility.

Millennium Management's flagship fund returned 15.2% net in Q2 with maximum drawdown of 2.1%, outperforming the MSCI World Index by 340 basis points. The firm is expanding its London office and hired three senior portfolio managers from Renaissance Technologies.""",
        
        "example_output": """{
  "geographic_focus": "Global with Asia-Pacific and European expansion",
  "people": [
    {
      "name": "John Smith",
      "current_company": "Citadel Asia",
      "current_title": "Managing Director",
      "previous_company": "Goldman Sachs",
      "movement_type": "hire",
      "location": "Hong Kong",
      "experience_years": "15",
      "expertise": "Equity Trading",
      "seniority_level": "senior"
    },
    {
      "name": "Lisa Chen",
      "current_company": "Dragon Capital Management",
      "current_title": "Founder/Portfolio Manager", 
      "previous_company": "JPMorgan",
      "movement_type": "launch",
      "location": "Unknown",
      "expertise": "Long/Short Equity",
      "seniority_level": "senior"
    }
  ],
  "firms": [
    {
      "name": "Dragon Capital Management",
      "firm_type": "Hedge Fund",
      "strategy": "Long/Short Equity",
      "geographic_focus": "Asian Markets",
      "aum": "200000000",
      "status": "newly_launched"
    },
    {
      "name": "Citadel Asia",
      "firm_type": "Hedge Fund",
      "location": "Hong Kong",
      "status": "expanding"
    },
    {
      "name": "Engineers Gate",
      "firm_type": "Hedge Fund", 
      "strategy": "Systematic Trading",
      "status": "operating"
    },
    {
      "name": "Millennium Management",
      "firm_type": "Hedge Fund",
      "status": "expanding",
      "expansion_location": "London"
    }
  ],
  "performance": [
    {
      "fund_name": "Engineers Gate",
      "metric_type": "aum",
      "value": "4200000000",
      "period": "Current",
      "date": "2025",
      "additional_info": "USD, systematic trading fund"
    },
    {
      "fund_name": "Engineers Gate",
      "metric_type": "return",
      "value": "12.3",
      "period": "YTD", 
      "date": "2025",
      "additional_info": "net return, percent"
    },
    {
      "fund_name": "Engineers Gate",
      "metric_type": "sharpe",
      "value": "1.8",
      "period": "Current",
      "date": "2025", 
      "additional_info": "improved from 1.2 previous year"
    },
    {
      "fund_name": "Engineers Gate",
      "metric_type": "drawdown",
      "value": "2.5",
      "period": "Q3",
      "date": "2025",
      "additional_info": "maximum drawdown below, percent"
    },
    {
      "fund_name": "Millennium Management",
      "metric_type": "return",
      "value": "15.2", 
      "period": "Q2",
      "date": "2025",
      "additional_info": "net return, flagship fund, percent"
    },
    {
      "fund_name": "Millennium Management",
      "metric_type": "drawdown",
      "value": "2.1",
      "period": "Q2", 
      "date": "2025",
      "additional_info": "maximum drawdown, percent"
    },
    {
      "fund_name": "Millennium Management",
      "metric_type": "alpha",
      "value": "340",
      "period": "Q2",
      "date": "2025",
      "benchmark": "MSCI World Index",
      "additional_info": "outperformance in basis points"
    }
  ]
}""",
        
        "output_format": """{
  "geographic_focus": "Primary geographic region or 'Global' if multiple regions",
  "people": [
    {
      "name": "Full Legal Name",
      "current_company": "Current Firm Name",
      "current_title": "Exact Job Title",
      "previous_company": "Former Firm (if mentioned)",
      "movement_type": "hire|promotion|launch|departure|appointment",
      "location": "City, Country or Region",
      "experience_years": "Number of years experience (if mentioned)",
      "expertise": "Area of specialization",
      "seniority_level": "junior|mid|senior|c_suite"
    }
  ],
  "firms": [
    {
      "name": "Exact Firm Name",
      "firm_type": "Hedge Fund|Investment Bank|Asset Manager|Private Equity|Family Office",
      "strategy": "Investment strategy or business line",
      "geographic_focus": "Geographic focus if mentioned",
      "aum": "Assets under management (numeric only)",
      "status": "launching|expanding|closing|operating|acquired"
    }
  ],
  "performance": [
    {
      "fund_name": "Exact Fund/Firm Name",
      "metric_type": "return|irr|sharpe|information_ratio|drawdown|alpha|beta|volatility|aum|tracking_error|correlation",
      "value": "numeric_value_only_no_units",
      "period": "YTD|Q1|Q2|Q3|Q4|1Y|3Y|5Y|ITD|Monthly|Current",
      "date": "YYYY or YYYY-MM",
      "benchmark": "Benchmark name if comparison mentioned",
      "additional_info": "Units, context, fund type, net/gross specification"
    }
  ]
}"""
    }

def build_extraction_prompt_with_cache(newsletter_text, cached_context):
    """Build enhanced extraction prompt using cached context for superior hedge fund intelligence"""
    
    prompt = f"""
{cached_context['system_instructions']}

CRITICAL EXTRACTION PROTOCOLS:
1. ZERO TOLERANCE for placeholder text - NEVER use "Full Name", "Company Name", "Exact Firm Name"
2. EXTRACT ONLY verified, specific names and firms explicitly mentioned in the text
3. PRIORITIZE senior-level movements (MD, VP, CIO, CEO, Portfolio Manager, Head of Trading)
4. CAPTURE numerical precision - exact percentages, dollar amounts, basis points
5. IDENTIFY industry context - hedge fund vs investment bank vs asset manager
6. DETERMINE seniority level from titles and context clues
7. EXTRACT geographic intelligence and market focus areas

ENHANCED TARGETING:
- Look for fund launches with specific AUM figures
- Identify performance attribution with benchmarks  
- Capture risk metrics in institutional context
- Track senior talent movements between major institutions
- Note expansion strategies and office openings
- Extract regulatory and compliance appointments

PROFESSIONAL TITLE MAPPING:
- Managing Director (MD) = senior level
- Vice President (VP) = senior level  
- Portfolio Manager (PM) = senior level
- Chief Investment Officer (CIO) = c_suite level
- Head of [Department] = senior level
- Analyst = junior/mid level
- Associate = mid level

EXAMPLE INPUT:
{cached_context['example_input']}

EXAMPLE OUTPUT:
{cached_context['example_output']}

REQUIRED OUTPUT FORMAT:
{cached_context['output_format']}

TARGET NEWSLETTER FOR ANALYSIS:
{newsletter_text}

EXTRACTION MANDATE: Extract ONLY concrete, verifiable information with complete names and specific institutions. If any field cannot be determined with certainty, omit that entry entirely. Focus on actionable intelligence for hedge fund industry tracking.

Return ONLY the JSON output with geographic_focus, people, firms, and performance arrays populated with verified data."""
    
    return prompt

# ENHANCED: Flexible file preprocessing with configurable options
def preprocess_newsletter_text(text, mode="balanced"):
    """
    Enhanced preprocessing with configurable modes for different file sizes and types
    
    Args:
        text: Input text to preprocess
        mode: Preprocessing intensity level
            - "minimal": Only basic cleaning, preserve most content
            - "balanced": Moderate filtering (default)
            - "aggressive": Heavy filtering for very noisy content
            - "none": Skip preprocessing entirely
    """
    import re
    
    if mode == "none":
        st.info("📄 **No preprocessing applied** - Processing raw content")
        return text
    
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
    if mode in ["balanced", "aggressive"]:
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
    if mode in ["balanced", "aggressive"]:
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # Standard URLs
            r'<https?://[^>]+>',  # URLs in angle brackets
            r'urldefense\.proofpoint\.com[^\s]*',  # Proofpoint URLs
            r'pardot\.withintelligence\.com[^\s]*',  # Tracking URLs
            r'jpmorgan\.email\.streetcontxt\.net[^\s]*'  # Email tracking
        ]
        
        for pattern in url_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 4: Remove email disclaimers and legal text (only in aggressive mode)
    if mode == "aggressive":
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
    if mode in ["balanced", "aggressive"]:
        html_patterns = [
            r'<[^>]+>',  # HTML tags
            r'&[a-zA-Z0-9#]+;',  # HTML entities
            r'\[cid:[^\]]+\]',  # Email embedded images
        ]
        
        for pattern in html_patterns:
            text = re.sub(pattern, '', text)
        
        # Only remove excessive formatting in aggressive mode
        if mode == "aggressive":
            text = re.sub(r'________________________________+', '', text)  # Email separators
            text = re.sub(r'\*\s*\|.*?\|\s*\*', '', text)  # Email table formatting
    
    # Step 6: Clean up excessive whitespace
    if mode in ["minimal", "balanced", "aggressive"]:
        # Remove multiple consecutive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Only filter lines in balanced/aggressive mode
        if mode in ["balanced", "aggressive"]:
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                # Keep line if it has meaningful content
                if re.search(r'[a-zA-Z].*[a-zA-Z]', line) and len(line.strip()) > 5:
                    # Clean up the line
                    line = re.sub(r'\s+', ' ', line.strip())  # Normalize whitespace
                    if line:
                        cleaned_lines.append(line)
            text = '\n'.join(cleaned_lines)
    
    # Step 7: Focus on hedge fund relevant content (only in aggressive mode)
    if mode == "aggressive":
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
            'appoints', 'appointed', 'hiring', 'hired', 'departure', 'departing', 'leaving', 'joining', 'joined', 'moved', 'moving', 'promoted', 'promotion', 'named', 'named as', 'becomes', 'became', 'takes over', 'steps down'
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
            elif len(para) > 100 and ('capital' in para_lower or 'management' in para_lower): # Keep longer paragraphs that might be relevant
                relevant_paragraphs.append(para)
        
        # If we didn't find enough relevant content, keep more of the original
        if len(relevant_paragraphs) < 3:
            relevant_paragraphs = paragraphs[:20] # Keep first 20 paragraphs as fallback
        
        text = '\n\n'.join(relevant_paragraphs)
    
    # Final cleanup
    text = text.strip()
    
    # Show cleaning results
    original_size = len(text) # Re-calculate original size after initial cleaning
    final_size = len(text) # Final size after aggressive filter
    reduction_pct = ((original_size - final_size) / original_size) * 100 if original_size > 0 else 0
    st.info(f"📝 **Text Preprocessing Complete** (Mode: {mode.title()})")
    st.write(f"• **Original size**: {original_size:,} characters")
    st.write(f"• **Processed size**: {final_size:,} characters")
    st.write(f"• **Reduction**: {reduction_pct:.1f}% content filtered")
    if mode == "aggressive":
        paragraphs_found = len(text.split('\n\n'))
        st.write(f"• **Relevant sections**: {paragraphs_found} found")
    
    return text

# ENHANCED: Better file type support with encoding detection
def load_file_content(uploaded_file):
    """
    Enhanced file loading with better encoding detection and file type support
    Args:
        uploaded_file: Streamlit uploaded file object
    Returns:
        tuple: (success: bool, content: str, error_message: str)
    """
    try:
        file_size = len(uploaded_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        st.info(f"📁 **File Details**: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Handle different file types
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            # Text file - try multiple encodings
            raw_data = uploaded_file.getvalue()
            # Try common encodings in order of preference
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
            content = None
            encoding_used = None
            for encoding in encodings:
                try:
                    content = raw_data.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return False, "", "Could not decode file. Please ensure it's a valid text file."
            st.success(f"✅ **Text file loaded** (encoding: {encoding_used})")
            return True, content, ""
        
        elif uploaded_file.type in ["application/pdf"] or uploaded_file.name.endswith('.pdf'):
            # PDF support would require additional libraries
            return False, "", "PDF files not yet supported. Please convert to .txt format."
        
        elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or uploaded_file.name.endswith(('.doc', '.docx')):
            # Word document support would require additional libraries
            return False, "", "Word documents not yet supported. Please save as .txt format."
        
        else:
            # Try to treat as text anyway
            try:
                raw_data = uploaded_file.getvalue()
                content = raw_data.decode('utf-8', errors='ignore')
                st.warning(f"⚠️ Unknown file type '{uploaded_file.type}'. Attempting to read as text...")
                return True, content, ""
            except Exception as e:
                return False, "", f"Unsupported file type: {uploaded_file.type}. Please use .txt files."
    
    except Exception as e:
        return False, "", f"Error reading file: {str(e)}"

def extract_single_chunk_safe(text, model):
    """Enhanced single chunk extraction with improved validation for hedge fund intelligence"""
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
        firms = result.get('firms', [])
        geographic_focus = result.get('geographic_focus', '')

        # Enhanced validation for hedge fund intelligence
        valid_people = []
        valid_performance = []
        
        # Validate people with enhanced structure
        for p in people:
            name = p.get('name', '').strip()
            current_company = p.get('current_company', '').strip()
            
            # Enhanced validation criteria
            if (name and current_company and 
                name.lower() not in ['full name', 'full legal name', 'name', 'person name', 'unknown'] and
                current_company.lower() not in ['company', 'current firm name', 'company name', 'firm name', 'unknown'] and
                len(name) > 2 and len(current_company) > 2 and
                not any(placeholder in name.lower() for placeholder in ['exact', 'sample', 'example']) and
                not any(placeholder in current_company.lower() for placeholder in ['exact', 'sample', 'example'])):
                
                # Map new structure to legacy structure for compatibility
                legacy_person = {
                    'name': name,
                    'company': current_company, # Map current_company to company for compatibility
                    'title': p.get('current_title', 'Unknown'),
                    'movement_type': p.get('movement_type', 'Unknown'),
                    'location': p.get('location', 'Unknown'),
                    # Preserve enhanced fields
                    'current_company': current_company,
                    'current_title': p.get('current_title', 'Unknown'),
                    'previous_company': p.get('previous_company', 'Unknown'),
                    'experience_years': p.get('experience_years', 'Unknown'),
                    'expertise': p.get('expertise', 'Unknown'),
                    'seniority_level': p.get('seniority_level', 'Unknown')
                }
                valid_people.append(legacy_person)

        # Validate firms with enhanced structure
        valid_firms = []
        for f in firms:
            name = f.get('name', '').strip()
            firm_type = f.get('firm_type', '').strip()
            if (name and firm_type and
                name.lower() not in ['exact firm name', 'firm name', 'unknown'] and
                len(name) > 2 and
                not any(placeholder in name.lower() for placeholder in ['exact', 'sample', 'example'])):
                
                # Map new structure to legacy structure for compatibility
                legacy_firm = {
                    'name': name,
                    'type': firm_type, # Map firm_type to type for compatibility
                    'location': f.get('location', 'Unknown'),
                    'strategy': f.get('strategy', 'Unknown'),
                    'aum': safe_get(f, 'aum', 'Unknown'), # Ensure AUM is handled safely
                    # Preserve enhanced fields
                    'firm_type': firm_type,
                    'geographic_focus': f.get('geographic_focus', 'Unknown'),
                    'headquarters': f.get('headquarters', 'Unknown'),
                    'founded': safe_get(f, 'founded', 'Unknown'),
                    'website': f.get('website', 'Unknown'),
                    'description': f.get('description', 'Unknown'),
                    'status': f.get('status', 'operating')
                }
                
                # Handle performance metrics within firms
                if f.get('performance_metrics'):
                    valid_performance.extend([
                        {**metric, 'fund_name': name} # Add firm name for context
                        for metric in f['performance_metrics']
                    ])
                
                valid_firms.append(legacy_firm)

        # Validate performance metrics and integrate with firms
        for p in performance:
            metric_type = p.get('metric_type', '').strip()
            value = safe_get(p, 'value', '').strip()
            fund_name = p.get('fund_name', '').strip() # Should be provided if directly in 'performance' array
            
            if (metric_type and value and fund_name and
                metric_type.lower() != 'metric_type' and value.lower() != 'numeric_value_only_no_units' and
                len(value) > 0 and len(fund_name) > 2):
                
                valid_performance.append({
                    'fund_name': fund_name,
                    'metric_type': metric_type,
                    'value': value,
                    'period': p.get('period', 'Unknown'),
                    'date': p.get('date', 'Unknown'),
                    'benchmark': p.get('benchmark', 'Unknown'),
                    'additional_info': p.get('additional_info', 'Unknown')
                })

        return valid_people, valid_firms, valid_performance

    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        st.warning("AI response might not be valid JSON. Please adjust prompt or input.")
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.code(response.text)
        return [], [], []
    except Exception as e:
        st.error(f"Error during AI extraction: {e}")
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e)
        return [], [], []


def aggregate_and_deduplicate(new_people, new_firms, new_employments, new_extractions):
    """
    Aggregates new extractions with existing session state data and deduplicates.
    Prioritizes retaining data already in session state if duplicates are found.
    
    Args:
        new_people, new_firms, new_employments, new_extractions: Lists of newly extracted data.
    """
    # People Deduplication
    existing_people_ids = {p['id'] for p in st.session_state.people}
    for person in new_people:
        if 'id' not in person:
            person['id'] = str(uuid.uuid4())
        if person['id'] not in existing_people_ids:
            st.session_state.people.append(person)
            existing_people_ids.add(person['id']) # Add new ID to set to prevent future duplicates in same run

    # Firm Deduplication
    existing_firms_ids = {f['id'] for f in st.session_state.firms}
    for firm in new_firms:
        if 'id' not in firm:
            firm['id'] = str(uuid.uuid4())
        if firm['id'] not in existing_firms_ids:
            st.session_state.firms.append(firm)
            existing_firms_ids.add(firm['id'])

    # Employment Deduplication (more complex, consider unique (person_id, company_name, start_date))
    # For simplicity, if an employment record with same person_id, company_name, and start_date exists, skip
    existing_employments_keys = {
        (emp.get('person_id'), emp.get('company_name'), emp.get('start_date')) 
        for emp in st.session_state.employments
    }
    for emp in new_employments:
        if 'id' not in emp:
            emp['id'] = str(uuid.uuid4())
        
        # Convert date objects in new_employments to date strings for comparison
        emp_start_date_str = emp['start_date'].strftime('%Y-%m-%d') if isinstance(emp.get('start_date'), date) else emp.get('start_date')
        emp_end_date_str = emp['end_date'].strftime('%Y-%m-%d') if isinstance(emp.get('end_date'), date) else emp.get('end_date')

        employment_key = (emp.get('person_id'), emp.get('company_name'), emp_start_date_str)
        if employment_key not in existing_employments_keys:
            # Ensure dates are date objects for session state storage
            if isinstance(emp.get('start_date'), str):
                try: emp['start_date'] = datetime.strptime(emp['start_date'], '%Y-%m-%d').date()
                except ValueError: pass # Keep as string if parsing fails
            if isinstance(emp.get('end_date'), str) and emp['end_date'] != 'Present':
                try: emp['end_date'] = datetime.strptime(emp['end_date'], '%Y-%m-%d').date()
                except ValueError: pass # Keep as string if parsing fails
            
            st.session_state.employments.append(emp)
            existing_employments_keys.add(employment_key)

    # Extractions Deduplication (by full content or ID if they have one)
    existing_extractions_content = {json.dumps(e, sort_keys=True) for e in st.session_state.all_extractions}
    for ext in new_extractions:
        # Add 'id' if missing for consistency
        if 'id' not in ext:
            ext['id'] = str(uuid.uuid4())
        
        # Check if an identical extraction (content-wise) already exists
        if json.dumps(ext, sort_keys=True) not in existing_extractions_content:
            st.session_state.all_extractions.append(ext)
            existing_extractions_content.add(json.dumps(ext, sort_keys=True))


# Helper to get unique values for dynamic inputs
def get_unique_values_from_session_state(table_name, field_name):
    """Extracts unique values for a given field from session state data."""
    if table_name == 'people':
        return sorted(list(set(safe_get(p, field_name) for p in st.session_state.people if safe_get(p, field_name) != 'Unknown')))
    elif table_name == 'firms':
        return sorted(list(set(safe_get(f, field_name) for f in st.session_state.firms if safe_get(f, field_name) != 'Unknown')))
    # Add more tables as needed
    return []

# --- Review System Functions ---
def get_review_time_remaining():
    """Calculates time remaining for review before auto-saving."""
    if st.session_state.review_start_time:
        elapsed_time = (datetime.now() - st.session_state.review_start_time).total_seconds()
        remaining = st.session_state.auto_save_timeout - elapsed_time
        return max(0, remaining)
    return 0

def auto_save_pending_reviews():
    """Moves pending review items to main data and saves."""
    saved_count = 0
    if st.session_state.pending_review_data:
        for item_type, data in st.session_state.pending_review_data:
            if item_type == 'person':
                st.session_state.people.append(data)
            elif item_type == 'firm':
                st.session_state.firms.append(data)
            elif item_type == 'employment':
                st.session_state.employments.append(data)
            elif item_type == 'extraction':
                st.session_state.all_extractions.append(data)
            saved_count += 1
        st.session_state.pending_review_data = [] # Clear the review queue
        st.session_state.review_start_time = None # Reset review timer
        save_data_async() # Trigger an asynchronous save after review handling [MODIFIED: Call async save]
    return saved_count


# --- Streamlit UI ---
initialize_session_state()

# Auto-save with review handling
current_time = datetime.now()
if 'last_auto_save' not in st.session_state:
    st.session_state.last_auto_save = current_time

# Auto-save every 30 seconds if there's data
time_since_save = (current_time - st.session_state.last_auto_save).total_seconds()
if time_since_save > 30 and (st.session_state.people or st.session_state.firms or st.session_state.all_extractions):
    save_data_async() # MODIFIED: Call the asynchronous save function
    st.session_state.last_auto_save = current_time

# Display save status in the sidebar (optional)
if st.session_state.save_status:
    st.sidebar.text(st.session_state.save_status)
    # Clear status after a short delay or next interaction if desired

# Handle review timeout
if st.session_state.pending_review_data and st.session_state.review_start_time:
    if get_review_time_remaining() <= 0:
        saved_count = auto_save_pending_reviews()
        if saved_count > 0:
            st.sidebar.success(f"⏰ Auto-saved {saved_count} items from review queue!")
            st.rerun()

# Auto-refresh for review interface using Streamlit's built-in mechanisms
if st.session_state.show_review_interface and st.session_state.pending_review_data:
    remaining = get_review_time_remaining()
    if remaining > 0:
        # Use JavaScript to refresh the page periodically when in review mode
        # This is a conceptual way to refresh, Streamlit's rerun handles this implicitly
        pass

# Main content goes here (e.g., file uploader, AI extraction, data display)
# ... (rest of your Streamlit app code) ...
