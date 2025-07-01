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
import logging
import threading
import queue

# Additional imports for enhanced export functionality
import zipfile
from io import BytesIO, StringIO

# Try to import openpyxl for Excel exports
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Try to import google.generativeai, handle if not available
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        if data is None:
            return default
        value = data.get(key, default)
        return value if value is not None and str(value).strip() != '' else default
    except Exception as e:
        logger.warning(f"Error in safe_get for key {key}: {e}")
        return default

# --- Enhanced File Loading with Encoding Detection ---
def load_file_content_enhanced(uploaded_file):
    """
    Enhanced file loading with robust encoding detection and error handling
    
    Returns:
        tuple: (success: bool, content: str, error_message: str, encoding_used: str)
    """
    try:
        file_size = len(uploaded_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Loading file: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Handle different file types
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            # Text file - try multiple encodings in order of preference
            raw_data = uploaded_file.getvalue()
            
            # Try common encodings in order
            encodings_to_try = [
                'utf-8',           # Standard UTF-8
                'utf-8-sig',       # UTF-8 with BOM
                'cp1252',          # Windows-1252 (common for Windows files)
                'latin1',          # ISO-8859-1
                'iso-8859-1',      # Alternative name for latin1
                'cp1251',          # Windows-1251 (Cyrillic)
                'ascii',           # Plain ASCII
                'utf-16',          # UTF-16 (less common)
                'utf-16le',        # UTF-16 Little Endian
                'utf-16be'         # UTF-16 Big Endian
            ]
            
            content = None
            encoding_used = None
            decode_errors = []
            
            for encoding in encodings_to_try:
                try:
                    content = raw_data.decode(encoding)
                    encoding_used = encoding
                    logger.info(f"Successfully decoded file with {encoding}")
                    break
                except UnicodeDecodeError as e:
                    decode_errors.append(f"{encoding}: {str(e)}")
                    continue
                except Exception as e:
                    decode_errors.append(f"{encoding}: {str(e)}")
                    continue
            
            if content is None:
                # Last resort: decode with 'replace' error handling
                try:
                    content = raw_data.decode('utf-8', errors='replace')
                    encoding_used = 'utf-8 (with replacements)'
                    logger.warning("Using UTF-8 with character replacements - some characters may be corrupted")
                except Exception as e:
                    error_msg = f"Could not decode file with any encoding. Tried: {', '.join([enc.split(':')[0] for enc in decode_errors])}"
                    logger.error(error_msg)
                    return False, "", error_msg, None
            
            # Validate content
            if not content or len(content.strip()) == 0:
                return False, "", "File appears to be empty after decoding", encoding_used
            
            # Check for binary content indicators
            binary_indicators = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
            if any(indicator in content for indicator in binary_indicators):
                logger.warning("File may contain binary data - proceeding with caution")
            
            logger.info(f"File loaded successfully: {len(content)} characters, encoding: {encoding_used}")
            return True, content, "", encoding_used
            
        elif uploaded_file.type in ["application/pdf"] or uploaded_file.name.endswith('.pdf'):
            return False, "", "PDF files not supported. Please convert to .txt format.", None
            
        elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or uploaded_file.name.endswith(('.doc', '.docx')):
            return False, "", "Word documents not supported. Please save as .txt format.", None
            
        else:
            # Unknown file type - try to treat as text with permissive decoding
            try:
                raw_data = uploaded_file.getvalue()
                content = raw_data.decode('utf-8', errors='replace')
                logger.warning(f"Unknown file type '{uploaded_file.type}'. Treating as text with UTF-8 replacement.")
                return True, content, f"Warning: Unknown file type '{uploaded_file.type}'. Some characters may be corrupted.", "utf-8 (permissive)"
            except Exception as e:
                return False, "", f"Unsupported file type: {uploaded_file.type}. Error: {str(e)}", None
    
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg)
        return False, "", error_msg, None

# Fallback function for compatibility
def load_file_content(uploaded_file):
    """Legacy function for compatibility - calls enhanced version"""
    try:
        success, content, error_msg, encoding_used = load_file_content_enhanced(uploaded_file)
        return success, content, error_msg
    except Exception as e:
        return False, "", f"Error loading file: {str(e)}"

def get_unique_values_from_session_state(table_name, field_name):
    """Get unique values for a field from session state data"""
    try:
        values = set()
        
        if table_name == 'people' and 'people' in st.session_state:
            for item in st.session_state.people:
                value = safe_get(item, field_name)
                if value and value != 'Unknown':
                    values.add(value)
        
        elif table_name == 'firms' and 'firms' in st.session_state:
            for item in st.session_state.firms:
                value = safe_get(item, field_name)
                if value and value != 'Unknown':
                    values.add(value)
        
        return list(values)
    except Exception as e:
        logger.warning(f"Error getting unique values for {table_name}.{field_name}: {e}")
        return []

def check_and_recover_stuck_processing():
    """Check for and recover from stuck background processing"""
    try:
        bg_proc = st.session_state.background_processing
        
        if not bg_proc['is_running']:
            return False
        
        # Check for timeout
        if 'last_activity' in bg_proc and bg_proc['last_activity']:
            time_since_activity = (datetime.now() - bg_proc['last_activity']).total_seconds()
            
            if time_since_activity > 300:  # 5 minutes timeout
                logger.warning(f"Processing timeout detected after {time_since_activity}s inactivity")
                
                # Force stop and save any results
                total_people = bg_proc.get('saved_people', 0) + len(bg_proc['results']['people'])
                total_metrics = bg_proc.get('saved_performance', 0) + len(bg_proc['results']['performance'])
                
                bg_proc.update({
                    'is_running': False,
                    'status_message': f'Auto-stopped due to timeout. Recovered {total_people} people, {total_metrics} metrics',
                    'errors': bg_proc['errors'] + ['Processing timeout - automatically recovered']
                })
                
                return True  # Indicate recovery occurred
        
        return False
    except Exception as e:
        logger.error(f"Error in recovery check: {e}")
        return False

def emergency_stop_processing():
    """Emergency stop function for stuck processing"""
    try:
        bg_proc = st.session_state.background_processing
        
        if bg_proc['is_running']:
            logger.warning("Emergency stop triggered")
            
            # Force stop
            total_people = bg_proc.get('saved_people', 0) + len(bg_proc['results']['people'])
            total_metrics = bg_proc.get('saved_performance', 0) + len(bg_proc['results']['performance'])
            
            bg_proc.update({
                'is_running': False,
                'status_message': f'Emergency stop. Found {total_people} people, {total_metrics} metrics',
                'errors': bg_proc['errors'] + ['Emergency stop triggered']
            })
            
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        return False

# --- MISSING FUNCTIONS - MOVED TO TOP ---

def handle_dynamic_input(field_name, current_value, table_name, context=""):
    """
    Enhanced dynamic input that prioritizes typing with suggestions
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
    
    # Cancel button outside form for better UX
    if st.button("❌ Cancel Edit", key="cancel_edit_firm_outside"):
        st.session_state.show_edit_firm_modal = False
        st.session_state.edit_firm_data = None
        st.rerun()
    
    # Cancel button outside form for better UX
    if st.button("❌ Cancel Edit", key="cancel_edit_person_outside"):
        st.session_state.show_edit_person_modal = False
        st.session_state.edit_person_data = None
        st.rerun()
        
        if len(existing_options) > 9:
            st.caption(f"... and {len(existing_options) - 9} more options available")
    
    # Return the user input (either typed or from session state if suggestion was clicked)
    return user_input.strip() if user_input else ""

def enhanced_global_search(query):
    """Enhanced global search function with better matching and debugging"""
    try:
        query_lower = query.lower().strip()
        
        if len(query_lower) < 2:
            return [], [], []
        
        matching_people = []
        matching_firms = []
        matching_metrics = []
        
        # Search people with enhanced matching
        for person in st.session_state.people:
            try:
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
            except Exception as e:
                logger.warning(f"Error searching person {person.get('id', 'unknown')}: {e}")
                continue
        
        # Search firms with enhanced matching  
        for firm in st.session_state.firms:
            try:
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
            except Exception as e:
                logger.warning(f"Error searching firm {firm.get('id', 'unknown')}: {e}")
                continue
        
        # Search performance metrics in firms
        for firm in st.session_state.firms:
            try:
                if firm.get('performance_metrics'):
                    for metric in firm['performance_metrics']:
                        try:
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
                        except Exception as e:
                            logger.warning(f"Error searching metric in firm {firm.get('name', 'unknown')}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Error searching metrics in firm {firm.get('id', 'unknown')}: {e}")
                continue
        
        return matching_people, matching_firms, matching_metrics
    
    except Exception as e:
        logger.error(f"Error in enhanced_global_search: {e}")
        return [], [], []

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
        
        return True
        
    except Exception as e:
        logger.error(f"Save error: {e}")
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
            logger.info(f"Loaded {len(people)} people from {PEOPLE_FILE}")
        
        # Load firms
        if FIRMS_FILE.exists():
            with open(FIRMS_FILE, 'r', encoding='utf-8') as f:
                firms = json.load(f)
            logger.info(f"Loaded {len(firms)} firms from {FIRMS_FILE}")
        
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
            logger.info(f"Loaded {len(employments)} employments from {EMPLOYMENTS_FILE}")
        
        # Load extractions
        if EXTRACTIONS_FILE.exists():
            with open(EXTRACTIONS_FILE, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
            logger.info(f"Loaded {len(extractions)} extractions from {EXTRACTIONS_FILE}")
        
        return people, firms, employments, extractions
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
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
            "strategy": "Long-only Growth Equity",
            "created_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "last_updated": (datetime.now() - timedelta(days=5)).isoformat(),
            "extraction_history": [{
                "extraction_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "source_type": "sample_data",
                "context_preview": "Sample data for demo purposes"
            }]
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
            "strategy": "Multi-Strategy Quantitative",
            "created_date": (datetime.now() - timedelta(days=15)).isoformat(),
            "last_updated": (datetime.now() - timedelta(days=2)).isoformat(),
            "extraction_history": [{
                "extraction_date": (datetime.now() - timedelta(days=15)).isoformat(),
                "source_type": "sample_data",
                "context_preview": "Sample data for demo purposes"
            }]
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
            "strategy": "Equity Long/Short",
            "created_date": (datetime.now() - timedelta(days=7)).isoformat(),
            "last_updated": datetime.now().isoformat(),
            "extraction_history": [{
                "extraction_date": (datetime.now() - timedelta(days=7)).isoformat(),
                "source_type": "sample_data",
                "context_preview": "Sample data for demo purposes"
            }]
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
            "created_date": (datetime.now() - timedelta(days=45)).isoformat(),
            "last_updated": (datetime.now() - timedelta(days=10)).isoformat(),
            "extraction_history": [{
                "extraction_date": (datetime.now() - timedelta(days=45)).isoformat(),
                "source_type": "sample_data",
                "context_preview": "Sample firm data for demo purposes"
            }],
            "performance_metrics": [
                {
                    "id": str(uuid.uuid4()),
                    "metric_type": "return",
                    "value": "12.5",
                    "period": "YTD",
                    "date": "2025",
                    "additional_info": "Net return",
                    "recorded_date": (datetime.now() - timedelta(days=10)).isoformat(),
                    "source_reliability": "sample_data"
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
            "created_date": (datetime.now() - timedelta(days=20)).isoformat(),
            "last_updated": (datetime.now() - timedelta(days=3)).isoformat(),
            "extraction_history": [{
                "extraction_date": (datetime.now() - timedelta(days=20)).isoformat(),
                "source_type": "sample_data",
                "context_preview": "Sample firm data for demo purposes"
            }],
            "performance_metrics": [
                {
                    "id": str(uuid.uuid4()),
                    "metric_type": "sharpe",
                    "value": "1.8",
                    "period": "Current",
                    "date": "2025",
                    "additional_info": "Improved from 1.2",
                    "recorded_date": (datetime.now() - timedelta(days=3)).isoformat(),
                    "source_reliability": "sample_data"
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
            "created_date": (datetime.now() - timedelta(days=5)).isoformat(),
            "last_updated": datetime.now().isoformat(),
            "extraction_history": [{
                "extraction_date": (datetime.now() - timedelta(days=5)).isoformat(),
                "source_type": "sample_data",
                "context_preview": "Sample firm data for demo purposes"
            }],
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
            "strategy": "Investment Banking",
            "created_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "extraction_context": {
                "extraction_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "source_type": "sample_data"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "person_id": li_id,
            "company_name": "Hillhouse Capital",
            "title": "Portfolio Manager",
            "start_date": date(2021, 9, 1),
            "end_date": None,
            "location": "Hong Kong",
            "strategy": "Growth Equity",
            "created_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "extraction_context": {
                "extraction_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "source_type": "sample_data"
            }
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
    
    # File processing preferences
    if 'preprocessing_mode' not in st.session_state:
        st.session_state.preprocessing_mode = "balanced"
    if 'chunk_size_preference' not in st.session_state:
        st.session_state.chunk_size_preference = "auto"
    
    # Review system
    if 'enable_review_mode' not in st.session_state:
        st.session_state.enable_review_mode = True
    if 'pending_review_data' not in st.session_state:
        st.session_state.pending_review_data = []
    if 'review_start_time' not in st.session_state:
        st.session_state.review_start_time = None
    if 'show_review_interface' not in st.session_state:
        st.session_state.show_review_interface = False
    if 'auto_save_timeout' not in st.session_state:
        st.session_state.auto_save_timeout = 180
    
    # BACKGROUND PROCESSING STATE with timeout tracking
    if 'background_processing' not in st.session_state:
        st.session_state.background_processing = {
            'is_running': False,
            'progress': 0,
            'total_chunks': 0,
            'current_chunk': 0,
            'status_message': '',
            'results': {'people': [], 'performance': []},
            'errors': [],
            'start_time': None,
            'last_activity': None,
            'saved_people': 0,
            'saved_performance': 0,
            'processing_id': None,
            'failed_chunks': []
        }

def check_and_recover_stuck_processing():
    """Check for and recover from stuck background processing"""
    bg_proc = st.session_state.background_processing
    
    if not bg_proc['is_running']:
        return
    
    # Check for timeout
    if 'last_activity' in bg_proc and bg_proc['last_activity']:
        time_since_activity = (datetime.now() - bg_proc['last_activity']).total_seconds()
        
        if time_since_activity > 300:  # 5 minutes timeout
            logger.warning(f"Processing timeout detected after {time_since_activity}s inactivity")
            
            # Force stop and save any results
            total_people = bg_proc.get('saved_people', 0) + len(bg_proc['results']['people'])
            total_metrics = bg_proc.get('saved_performance', 0) + len(bg_proc['results']['performance'])
            
            bg_proc.update({
                'is_running': False,
                'status_message': f'Auto-stopped due to timeout. Recovered {total_people} people, {total_metrics} metrics',
                'errors': bg_proc['errors'] + ['Processing timeout - automatically recovered']
            })
            
            # Try to save any remaining results
            if bg_proc['results']['people'] or bg_proc['results']['performance']:
                if st.session_state.enable_review_mode:
                    add_to_review_queue(
                        bg_proc['results']['people'], 
                        bg_proc['results']['performance'], 
                        "Auto-recovered from timeout"
                    )
                else:
                    try:
                        saved_p, saved_perf = save_approved_extractions(
                            bg_proc['results']['people'], 
                            bg_proc['results']['performance']
                        )
                        bg_proc['saved_people'] += saved_p
                        bg_proc['saved_performance'] += saved_perf
                        logger.info(f"Auto-recovery saved {saved_p} people, {saved_perf} metrics")
                    except Exception as e:
                        logger.error(f"Auto-recovery save failed: {e}")
            
            return True  # Indicate recovery occurred
    
    return False

def emergency_stop_processing():
    """Emergency stop function for stuck processing"""
    bg_proc = st.session_state.background_processing
    
    if bg_proc['is_running']:
        logger.warning("Emergency stop triggered")
        
        # Try to save current results before stopping
        results_saved = False
        if bg_proc['results']['people'] or bg_proc['results']['performance']:
            try:
                if st.session_state.enable_review_mode:
                    add_to_review_queue(
                        bg_proc['results']['people'], 
                        bg_proc['results']['performance'], 
                        "Emergency Stop Recovery"
                    )
                    results_saved = True
                else:
                    saved_p, saved_perf = save_approved_extractions(
                        bg_proc['results']['people'], 
                        bg_proc['results']['performance']
                    )
                    bg_proc['saved_people'] += saved_p
                    bg_proc['saved_performance'] += saved_perf
                    results_saved = True
                    logger.info(f"Emergency save: {saved_p} people, {saved_perf} metrics")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
        
        # Force stop
        total_people = bg_proc.get('saved_people', 0) + len(bg_proc['results']['people'])
        total_metrics = bg_proc.get('saved_performance', 0) + len(bg_proc['results']['performance'])
        
        bg_proc.update({
            'is_running': False,
            'status_message': f'Emergency stop. {"Results saved" if results_saved else "No results to save"}. Found {total_people} people, {total_metrics} metrics',
            'errors': bg_proc['errors'] + ['Emergency stop triggered']
        })
        
        return True
    
    return False

def get_unique_values_from_session_state(table_name, field_name):
    """Get unique values for a field from session state data"""
    try:
        values = set()
        
        if table_name == 'people' and 'people' in st.session_state:
            for item in st.session_state.people:
                value = safe_get(item, field_name)
                if value and value != 'Unknown':
                    values.add(value)
        
        elif table_name == 'firms' and 'firms' in st.session_state:
            for item in st.session_state.firms:
                value = safe_get(item, field_name)
                if value and value != 'Unknown':
                    values.add(value)
        
        return list(values)
    except Exception as e:
        logger.warning(f"Error getting unique values for {table_name}.{field_name}: {e}")
        return []

# --- AI Setup with Enhanced Model Support ---
@st.cache_resource
def setup_gemini(api_key, model_id="gemini-1.5-flash"):
    """Setup Gemini AI model safely with model selection"""
    if not GENAI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        model.model_id = model_id
        return model
    except Exception as e:
        logger.error(f"AI setup failed: {e}")
        return None

def get_model_rate_limits(model_id):
    """Get rate limits for different Gemini models with proper 2.0/2.5 handling"""
    rate_limits = {
        "gemini-1.5-flash": {"requests_per_minute": 15, "delay": 4},
        "gemini-1.5-flash-latest": {"requests_per_minute": 15, "delay": 4},
        "gemini-1.5-flash-8b": {"requests_per_minute": 15, "delay": 4},
        "gemini-1.5-pro": {"requests_per_minute": 2, "delay": 30},
        "gemini-1.5-pro-latest": {"requests_per_minute": 2, "delay": 30},
        # Conservative limits for 2.0 models (unclear official limits)
        "gemini-2.0-flash-thinking-exp": {"requests_per_minute": 5, "delay": 12},  # Actual 2.0 
        "gemini-2.0-flash-exp": {"requests_per_minute": 4, "delay": 15},  # Experimental 2.0
        "gemini-2.5-flash-exp": {"requests_per_minute": 3, "delay": 20},  # Very conservative for 2.5
        "gemini-exp-1114": {"requests_per_minute": 2, "delay": 30},
        "gemini-exp-1121": {"requests_per_minute": 2, "delay": 30}
    }
    
    # Get base limits
    base_limits = rate_limits.get(model_id, {"requests_per_minute": 5, "delay": 12})
    
    # Special handling for 2.0+ models which can be unstable
    if any(version in model_id for version in ["2.0", "2.5"]):
        base_limits["delay"] = max(base_limits["delay"], 12)  # Minimum 12s delay
        base_limits["max_retries"] = 3
        base_limits["timeout_seconds"] = 90  # Longer timeout for newer models
        base_limits["experimental"] = True
    else:
        base_limits["timeout_seconds"] = 60
        base_limits["experimental"] = False
    
    return base_limits

@st.cache_data(ttl=3600)
def create_cached_context():
    """Create cached context for hedge fund extraction with optimized prompts"""
    return {
        "system_instructions": """You are an expert financial analyst specializing in the hedge fund industry. Your task is to meticulously analyze the following text and extract key intelligence about hedge funds, investment banks, asset managers, private equity firms, and related financial institutions.

CORE EXTRACTION TARGETS:
1. PEOPLE: All individuals in professional contexts (current employees, new hires, departures, promotions, launches, appointments)
2. FIRMS: Hedge funds, investment banks, asset managers, family offices, private equity, sovereign wealth funds
3. PERFORMANCE DATA: Returns, risk metrics, AUM figures, fund performance, benchmarks
4. MOVEMENTS: Job changes, fund launches, firm transitions, strategic shifts""",
        
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
    """Build enhanced extraction prompt using cached context"""
    prompt = f"""
{cached_context['system_instructions']}

TARGET NEWSLETTER FOR ANALYSIS:
{newsletter_text}

REQUIRED OUTPUT FORMAT:
{cached_context['output_format']}

Return ONLY the JSON output with geographic_focus, people, firms, and performance arrays populated with verified data."""
    
    return prompt

def preprocess_newsletter_text(text, mode="balanced"):
    """Enhanced preprocessing with configurable modes"""
    import re
    
    if mode == "none":
        return text
    
    original_size = len(text)
    
    # Basic cleaning
    if mode in ["balanced", "aggressive"]:
        # Remove email headers
        email_header_patterns = [
            r'From:\s*.*?\n', r'To:\s*.*?\n', r'Sent:\s*.*?\n',
            r'Subject:\s*.*?\n', r'Date:\s*.*?\n'
        ]
        for pattern in email_header_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove URLs
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'<https?://[^>]+>'
        ]
        for pattern in url_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Aggressive cleaning
    if mode == "aggressive":
        # Remove HTML and disclaimers
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        
        # Remove legal disclaimers
        disclaimer_patterns = [
            r'This message is confidential.*?(?=\n\n|\Z)',
            r'©.*?All rights reserved.*?(?=\n\n|\Z)',
            r'Unsubscribe.*?(?=\n\n|\Z)'
        ]
        for pattern in disclaimer_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()
    
    final_size = len(text)
    reduction_pct = ((original_size - final_size) / original_size) * 100 if original_size > 0 else 0
    
    logger.info(f"Text preprocessing complete. Mode: {mode}, Reduction: {reduction_pct:.1f}%")
    
    return text

def extract_single_chunk_safe(text, model):
    """Enhanced single chunk extraction with timeout and improved error handling"""
    try:
        cached_context = create_cached_context()
        prompt = build_extraction_prompt_with_cache(text, cached_context)
        
        # Get model-specific timeout
        rate_limits = get_model_rate_limits(model.model_id)
        timeout = rate_limits.get('timeout_seconds', 30)
        
        # Set generation config for more reliable responses
        generation_config = {
            'temperature': 0.1,  # Lower temperature for more consistent results
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
        
        # Add timeout for Flash 2.0
        if "2.0-flash" in model.model_id:
            generation_config['candidate_count'] = 1  # Single candidate for stability
        
        logger.info(f"Extracting with model {model.model_id}, timeout: {timeout}s, chunk size: {len(text)}")
        
        # Generate content with timeout handling
        try:
            response = model.generate_content(prompt, generation_config=generation_config)
            
            if not response or not response.text:
                logger.warning("Empty response from AI model")
                return [], []
            
            # Check for content filtering
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason'):
                    logger.warning(f"Content blocked: {response.prompt_feedback.block_reason}")
                    return [], []
            
        except Exception as generation_error:
            logger.error(f"Generation failed: {generation_error}")
            if "timeout" in str(generation_error).lower():
                raise TimeoutError(f"Model request timed out after {timeout}s")
            else:
                raise generation_error
        
        # Parse JSON safely with better error handling
        response_text = response.text.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            logger.warning("No valid JSON found in AI response")
            # Try to find JSON in different formats
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                logger.warning("No JSON found even with regex")
                return [], []
        else:
            json_text = response_text[json_start:json_end]
        
        try:
            result = json.loads(json_text)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing failed: {json_error}")
            logger.error(f"JSON text: {json_text[:500]}...")
            return [], []
        
        people = result.get('people', [])
        performance = result.get('performance', [])
        
        # Enhanced validation with better logging
        valid_people = []
        valid_performance = []
        
        for i, p in enumerate(people):
            try:
                name = safe_get(p, 'name', '').strip()
                current_company = safe_get(p, 'current_company', '').strip()
                
                if (name and current_company and 
                    name.lower() not in ['full name', 'full legal name', 'name', 'unknown'] and
                    current_company.lower() not in ['company', 'current firm name', 'firm name', 'unknown'] and
                    len(name) > 2 and len(current_company) > 2):
                    
                    # Map to legacy structure for compatibility
                    legacy_person = {
                        'name': name,
                        'company': current_company,
                        'title': safe_get(p, 'current_title', 'Unknown'),
                        'movement_type': safe_get(p, 'movement_type', 'Unknown'),
                        'location': safe_get(p, 'location', 'Unknown'),
                        'current_company': current_company,
                        'current_title': safe_get(p, 'current_title', 'Unknown'),
                        'previous_company': safe_get(p, 'previous_company', 'Unknown'),
                        'experience_years': safe_get(p, 'experience_years', 'Unknown'),
                        'expertise': safe_get(p, 'expertise', 'Unknown'),
                        'seniority_level': safe_get(p, 'seniority_level', 'Unknown')
                    }
                    valid_people.append(legacy_person)
                else:
                    logger.debug(f"Person {i} failed validation: name='{name}', company='{current_company}'")
            except Exception as person_error:
                logger.warning(f"Error processing person {i}: {person_error}")
                continue
        
        for i, p in enumerate(performance):
            try:
                fund_name = safe_get(p, 'fund_name', '').strip()
                metric_type = safe_get(p, 'metric_type', '').strip()
                value = safe_get(p, 'value', '').strip()
                
                if (fund_name and metric_type and value and
                    fund_name.lower() not in ['fund name', 'exact fund name', 'unknown'] and
                    metric_type.lower() not in ['metric', 'metric type', 'unknown'] and
                    value.lower() not in ['value', 'numeric value only', 'unknown']):
                    valid_performance.append(p)
                else:
                    logger.debug(f"Performance {i} failed validation: fund='{fund_name}', metric='{metric_type}', value='{value}'")
            except Exception as perf_error:
                logger.warning(f"Error processing performance {i}: {perf_error}")
                continue
        
        logger.info(f"Extraction complete: {len(valid_people)} people, {len(valid_performance)} performance metrics")
        return valid_people, valid_performance
        
    except TimeoutError as timeout_error:
        logger.error(f"Extraction timed out: {timeout_error}")
        raise timeout_error
    except json.JSONDecodeError as json_error:
        logger.error(f"JSON parsing failed: {json_error}")
        return [], []
    except Exception as e:
        logger.error(f"Enhanced extraction failed: {e}")
        return [], []

# --- BACKGROUND PROCESSING FUNCTIONS ---
def start_background_extraction(text, model, preprocessing_mode, chunk_size_mode):
    """Start background extraction process with improved error handling and auto-save"""
    st.session_state.background_processing = {
        'is_running': True,
        'progress': 0,
        'total_chunks': 1,
        'current_chunk': 0,
        'status_message': 'Starting extraction...',
        'results': {'people': [], 'performance': []},
        'errors': [],
        'start_time': datetime.now(),
        'last_activity': datetime.now(),
        'saved_people': 0,
        'saved_performance': 0,
        'processing_id': str(uuid.uuid4())
    }
    
    try:
        # Preprocess text
        cleaned_text = preprocess_newsletter_text(text, preprocessing_mode)
        
        # Determine chunking
        chunk_sizes = {
            "small": 10000, "medium": 20000, "large": 35000, "xlarge": 50000,
            "auto": min(max(len(cleaned_text) // 50, 15000), 35000)
        }
        chunk_size = chunk_sizes.get(chunk_size_mode, 20000)
        
        # Create chunks
        chunks = []
        current_pos = 0
        while current_pos < len(cleaned_text):
            end_pos = min(current_pos + chunk_size, len(cleaned_text))
            if end_pos < len(cleaned_text):
                search_start = max(end_pos - 500, current_pos)
                para_break = cleaned_text.rfind('\n\n', search_start, end_pos)
                if para_break > current_pos:
                    end_pos = para_break + 2
            
            chunk = cleaned_text[current_pos:end_pos].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
            current_pos = end_pos
        
        st.session_state.background_processing['total_chunks'] = len(chunks)
        logger.info(f"Starting background extraction: {len(chunks)} chunks, model: {model.model_id}")
        
        # Process chunks with incremental saving
        rate_limits = get_model_rate_limits(model.model_id)
        delay = rate_limits['delay']
        
        all_people = []
        all_performance = []
        failed_chunks = []
        consecutive_failures = 0
        
        for i, chunk in enumerate(chunks):
            # Check if processing should stop
            if not st.session_state.background_processing['is_running']:
                logger.info(f"Processing stopped by user at chunk {i+1}")
                break
            
            # Update activity timestamp
            st.session_state.background_processing['last_activity'] = datetime.now()
            
            # Update progress
            st.session_state.background_processing.update({
                'current_chunk': i + 1,
                'progress': int(((i + 1) / len(chunks)) * 100),
                'status_message': f'Processing chunk {i + 1}/{len(chunks)}...'
            })
            
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)} (size: {len(chunk)} chars)")
                people, performance = extract_single_chunk_safe(chunk, model)
                
                if people or performance:
                    all_people.extend(people)
                    all_performance.extend(performance)
                    consecutive_failures = 0  # Reset failure counter
                    
                    # Auto-save every 10 chunks or if we have 50+ items
                    if (i + 1) % 10 == 0 or len(all_people) >= 50:
                        if not st.session_state.enable_review_mode:
                            # Direct save
                            saved_p, saved_perf = save_approved_extractions(all_people, all_performance)
                            st.session_state.background_processing['saved_people'] += saved_p
                            st.session_state.background_processing['saved_performance'] += saved_perf
                            
                            # Clear saved items from memory
                            all_people = []
                            all_performance = []
                            
                            logger.info(f"Auto-saved {saved_p} people, {saved_perf} metrics at chunk {i+1}")
                    
                    logger.info(f"Chunk {i+1} success: {len(people)} people, {len(performance)} metrics")
                else:
                    consecutive_failures += 1
                    logger.warning(f"Chunk {i+1}: No results found")
                
                # Store results
                st.session_state.background_processing['results'] = {
                    'people': all_people,
                    'performance': all_performance
                }
                
                # Rate limiting with exponential backoff on failures
                if consecutive_failures >= 3:
                    actual_delay = min(delay * (1.5 ** consecutive_failures), delay * 5)
                    logger.warning(f"Multiple failures detected, increasing delay to {actual_delay}s")
                else:
                    actual_delay = delay
                
                if i < len(chunks) - 1:  # Don't delay after last chunk
                    time.sleep(actual_delay)
                
                # Emergency stop if too many consecutive failures
                if consecutive_failures >= 10:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping processing")
                    st.session_state.background_processing['errors'].append(f"Stopped due to {consecutive_failures} consecutive failures")
                    break
                    
            except TimeoutError as timeout_error:
                consecutive_failures += 1
                error_msg = f"Chunk {i + 1} timed out: {str(timeout_error)}"
                st.session_state.background_processing['errors'].append(error_msg)
                failed_chunks.append(i + 1)
                logger.error(error_msg)
                
                # For timeouts, wait longer before retrying
                time.sleep(delay * 2)
                
                if consecutive_failures >= 5:
                    logger.error(f"Too many timeouts ({consecutive_failures}), stopping processing")
                    break
                    
            except Exception as e:
                consecutive_failures += 1
                error_msg = f"Chunk {i + 1} failed: {str(e)}"
                st.session_state.background_processing['errors'].append(error_msg)
                failed_chunks.append(i + 1)
                logger.error(error_msg)
                
                # Handle specific errors
                if "rate" in str(e).lower() or "quota" in str(e).lower():
                    logger.warning(f"Rate limit hit at chunk {i+1}, waiting longer...")
                    time.sleep(delay * 3)  # Wait 3x longer on rate limits
                elif "503" in str(e) or "502" in str(e):
                    logger.warning(f"Server error at chunk {i+1}, waiting...")
                    time.sleep(delay * 2)
                elif consecutive_failures >= 5:
                    logger.error(f"Too many failures ({consecutive_failures}), stopping processing")
                    break
        
        # Final save of any remaining items
        if all_people or all_performance:
            if st.session_state.enable_review_mode:
                # Add to review queue
                add_to_review_queue(all_people, all_performance, f"Background Extraction ({len(chunks)} chunks)")
                logger.info(f"Added {len(all_people)} people, {len(all_performance)} metrics to review queue")
            else:
                # Direct save
                saved_p, saved_perf = save_approved_extractions(all_people, all_performance)
                st.session_state.background_processing['saved_people'] += saved_p
                st.session_state.background_processing['saved_performance'] += saved_perf
                logger.info(f"Final save: {saved_p} people, {saved_perf} metrics")
        
        # Complete processing
        total_found = st.session_state.background_processing['saved_people'] + len(st.session_state.background_processing['results']['people'])
        total_metrics = st.session_state.background_processing['saved_performance'] + len(st.session_state.background_processing['results']['performance'])
        
        st.session_state.background_processing.update({
            'is_running': False,
            'status_message': f'Completed! Found {total_found} people, {total_metrics} metrics',
            'progress': 100,
            'failed_chunks': failed_chunks
        })
        
        logger.info(f"Background extraction completed: {total_found} people, {total_metrics} metrics, {len(failed_chunks)} failed chunks")
        
    except Exception as e:
        st.session_state.background_processing.update({
            'is_running': False,
            'status_message': f'Failed: {str(e)}',
            'errors': [str(e)]
        })
        logger.error(f"Background extraction failed: {e}")

def display_background_processing_widget():
    """Display compact background processing widget with timeout detection"""
    bg_proc = st.session_state.background_processing
    
    if not bg_proc['is_running'] and bg_proc['progress'] == 0:
        return
    
    # Check for timeout/stuck processing
    if bg_proc['is_running'] and 'last_activity' in bg_proc:
        time_since_activity = (datetime.now() - bg_proc['last_activity']).total_seconds()
        if time_since_activity > 300:  # 5 minutes timeout
            st.session_state.background_processing['is_running'] = False
            st.session_state.background_processing['status_message'] = 'Timeout - Processing stopped automatically'
            st.session_state.background_processing['errors'].append('Processing timeout after 5 minutes of inactivity')
    
    with st.container():
        if bg_proc['is_running']:
            progress = bg_proc['progress']
            current = bg_proc['current_chunk']
            total = bg_proc['total_chunks']
            
            # Calculate elapsed time properly
            if 'start_time' in bg_proc and bg_proc['start_time']:
                elapsed_seconds = (datetime.now() - bg_proc['start_time']).total_seconds()
                if elapsed_seconds < 60:
                    time_str = f"{int(elapsed_seconds)}s"
                elif elapsed_seconds < 3600:
                    time_str = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60)}s"
                else:
                    time_str = f"{int(elapsed_seconds // 3600)}h {int((elapsed_seconds % 3600) // 60)}m"
            else:
                time_str = "0s"
            
            # Enhanced progress display
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.progress(progress / 100, text=f"🤖 AI Extraction: {progress}% • Chunk {current}/{total} • {time_str}")
            with col2:
                if st.button("⏹️ Stop", key="stop_background", help="Stop processing"):
                    st.session_state.background_processing['is_running'] = False
                    st.rerun()
            with col3:
                people_count = bg_proc.get('saved_people', 0) + len(bg_proc['results']['people'])
                metrics_count = bg_proc.get('saved_performance', 0) + len(bg_proc['results']['performance'])
                st.metric("Found", f"{people_count}P/{metrics_count}M", label_visibility="collapsed")
            
            # Show live stats
            if bg_proc.get('saved_people', 0) > 0 or bg_proc.get('saved_performance', 0) > 0:
                st.caption(f"💾 Saved: {bg_proc.get('saved_people', 0)} people, {bg_proc.get('saved_performance', 0)} metrics")
        
        else:
            # Show completion status
            results = bg_proc['results']
            errors = bg_proc['errors']
            failed_chunks = bg_proc.get('failed_chunks', [])
            
            if results['people'] or results['performance'] or bg_proc.get('saved_people', 0) > 0:
                total_people = bg_proc.get('saved_people', 0) + len(results['people'])
                total_metrics = bg_proc.get('saved_performance', 0) + len(results['performance'])
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if failed_chunks:
                        st.warning(f"✅ Extraction complete! {total_people} people, {total_metrics} metrics • {len(failed_chunks)} chunks failed")
                    else:
                        st.success(f"✅ Extraction complete! {total_people} people, {total_metrics} metrics")
                with col2:
                    if results['people'] or results['performance']:  # Only show if there's unsaved data
                        if st.button("📋 Review", key="review_results"):
                            source_info = f"Background Extraction ({bg_proc.get('processing_id', 'unknown')[:8]})"
                            add_to_review_queue(results['people'], results['performance'], source_info)
                            # Clear results after adding to review
                            st.session_state.background_processing['results'] = {'people': [], 'performance': []}
                            st.rerun()
                    else:
                        st.success("💾 All Saved")
                with col3:
                    if st.button("❌ Dismiss", key="dismiss_results"):
                        st.session_state.background_processing = {
                            'is_running': False, 'progress': 0, 'total_chunks': 0, 'current_chunk': 0,
                            'status_message': '', 'results': {'people': [], 'performance': []}, 'errors': [], 'start_time': None
                        }
                        st.rerun()
                
                # Show detailed stats in expander
                if failed_chunks or errors:
                    with st.expander("📊 Processing Details", expanded=False):
                        if failed_chunks:
                            st.error(f"Failed chunks: {', '.join(map(str, failed_chunks))}")
                        if errors:
                            st.error("Recent errors:")
                            for error in errors[-3:]:  # Show last 3 errors
                                st.code(error)
            
            elif errors:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.error(f"❌ Extraction failed: {errors[-1] if errors else 'Unknown error'}")
                with col2:
                    if st.button("❌ Dismiss", key="dismiss_error"):
                        st.session_state.background_processing = {
                            'is_running': False, 'progress': 0, 'total_chunks': 0, 'current_chunk': 0,
                            'status_message': '', 'results': {'people': [], 'performance': []}, 'errors': [], 'start_time': None
                        }
                        st.rerun()
            
            else:
                # No results found
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("ℹ️ No results found in processed text")
                with col2:
                    if st.button("❌ Dismiss", key="dismiss_no_results"):
                        st.session_state.background_processing = {
                            'is_running': False, 'progress': 0, 'total_chunks': 0, 'current_chunk': 0,
                            'status_message': '', 'results': {'people': [], 'performance': []}, 'errors': [], 'start_time': None
                        }
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

def get_person_performance_metrics(person_id):
    """Get performance metrics related to a specific person"""
    person = get_person_by_id(person_id)
    if not person:
        return []
    
    person_company = safe_get(person, 'current_company_name').lower()
    related_metrics = []
    
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

# --- PAGINATION HELPERS ---
def paginate_data(data, page, items_per_page=10):
    """Paginate data and return current page items and pagination info"""
    total_items = len(data)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
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

# --- REVIEW SYSTEM FUNCTIONS ---
def add_to_review_queue(people_extractions, performance_extractions, source_info="Manual extraction"):
    """Add extracted data to review queue"""
    review_item = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now(),
        'source': source_info,
        'people': people_extractions,
        'performance': performance_extractions,
        'status': 'pending',
        'reviewed_people': [],
        'reviewed_performance': []
    }
    
    st.session_state.pending_review_data.append(review_item)
    st.session_state.review_start_time = datetime.now()
    st.session_state.show_review_interface = True
    
    return review_item['id']

def get_review_time_remaining():
    """Get remaining time before auto-save (in seconds)"""
    if not st.session_state.review_start_time:
        return st.session_state.auto_save_timeout
    
    elapsed = (datetime.now() - st.session_state.review_start_time).total_seconds()
    remaining = max(0, st.session_state.auto_save_timeout - elapsed)
    return remaining

def display_review_interface():
    """Display the review interface for pending extractions"""
    if not st.session_state.pending_review_data:
        return
    
    st.markdown("---")
    st.header("📋 Review Extracted Data")
    
    # Timer display and controls
    remaining_time = get_review_time_remaining()
    if remaining_time > 0:
        minutes_left = int(remaining_time // 60)
        seconds_left = int(remaining_time % 60)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"⏱️ **Auto-save in**: {minutes_left}m {seconds_left}s")
        with col2:
            if st.button("💾 Save All Now", use_container_width=True, key="save_all_now_button"):
                saved_count = 0
                for review_item in st.session_state.pending_review_data:
                    approved_people, approved_performance = approve_all_in_review(review_item['id'])
                    source_context = f"Review batch approved: {review_item['source']}. Timestamp: {review_item['timestamp']}"
                    people_saved, perf_saved = save_approved_extractions(approved_people, approved_performance, source_context)
                    saved_count += people_saved
                
                st.session_state.pending_review_data = []
                st.session_state.show_review_interface = False
                st.success(f"✅ Saved {saved_count} items to database!")
                st.rerun()
        with col3:
            if st.button("❌ Cancel Review", use_container_width=True, key="cancel_review_button"):
                st.session_state.pending_review_data = []
                st.session_state.show_review_interface = False
                st.rerun()
    
    # Display review items (simplified for space)
    for i, review_item in enumerate(st.session_state.pending_review_data):
        st.markdown(f"### 📦 Batch {i+1}: {review_item['source']}")
        
        # People review
        if review_item['people']:
            st.markdown("#### 👥 People Found")
            for j, person in enumerate(review_item['people'][:5]):  # Show first 5
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{safe_get(person, 'name')}**")
                    st.caption(f"{safe_get(person, 'title')} at {safe_get(person, 'company')}")
                with col2:
                    st.caption(f"📍 {safe_get(person, 'location')}")
                with col3:
                    person_id = f"person_{review_item['id']}_{j}"
                    is_approved = person in review_item.get('reviewed_people', [])
                    if st.button("✅" if not is_approved else "❌", key=person_id):
                        if is_approved:
                            review_item['reviewed_people'] = [p for p in review_item['reviewed_people'] if p != person]
                        else:
                            review_item['reviewed_people'].append(person)
                        st.rerun()
        
        # Performance review
        if review_item['performance']:
            st.markdown("#### 📊 Performance Metrics")
            for j, metric in enumerate(review_item['performance'][:3]):  # Show first 3
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{safe_get(metric, 'fund_name')}**")
                    st.caption(f"{safe_get(metric, 'metric_type')}: {safe_get(metric, 'value')}")
                with col2:
                    st.caption(f"📅 {safe_get(metric, 'period')}")
                with col3:
                    metric_id = f"metric_{review_item['id']}_{j}"
                    is_approved = metric in review_item.get('reviewed_performance', [])
                    if st.button("✅" if not is_approved else "❌", key=metric_id):
                        if is_approved:
                            review_item['reviewed_performance'] = [m for m in review_item['reviewed_performance'] if m != metric]
                        else:
                            review_item['reviewed_performance'].append(metric)
                        st.rerun()

def approve_all_in_review(review_id):
    """Approve all items in a specific review"""
    review_item = next((r for r in st.session_state.pending_review_data if r['id'] == review_id), None)
    if not review_item:
        return [], []
    
    review_item['reviewed_people'] = review_item['people'].copy()
    review_item['reviewed_performance'] = review_item['performance'].copy()
    review_item['status'] = 'approved'
    
    return review_item['reviewed_people'], review_item['reviewed_performance']

def save_approved_extractions(approved_people, approved_performance, source_context=""):
    """Save approved extractions to main database with historical context"""
    saved_people = 0
    saved_performance = 0
    
    # Create source metadata
    source_metadata = {
        "extraction_date": datetime.now().isoformat(),
        "source_type": "newsletter_extraction", 
        "context_preview": source_context[:500] + "..." if len(source_context) > 500 else source_context,
        "total_people_in_batch": len(approved_people),
        "total_metrics_in_batch": len(approved_performance)
    }
    
    # Process people with historical context
    for person_data in approved_people:
        new_person_id = str(uuid.uuid4())
        company_name = person_data.get('current_company') or person_data.get('company', 'Unknown')
        title = person_data.get('current_title') or person_data.get('title', 'Unknown')
        
        # Create person with historical notes
        new_person = {
            "id": new_person_id,
            "name": safe_get(person_data, 'name'),
            "current_title": title,
            "current_company_name": company_name,
            "location": safe_get(person_data, 'location'),
            "email": "",
            "linkedin_profile_url": "",
            "phone": "",
            "education": "",
            "expertise": safe_get(person_data, 'expertise'),
            "aum_managed": "",
            "strategy": safe_get(person_data, 'expertise', 'Unknown'),
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            # Historical context
            "extraction_history": [{
                **source_metadata,
                "movement_type": safe_get(person_data, 'movement_type'),
                "previous_company": safe_get(person_data, 'previous_company'),
                "seniority_level": safe_get(person_data, 'seniority_level'),
                "original_extraction": person_data
            }]
        }
        
        st.session_state.people.append(new_person)
        
        # Add firm if doesn't exist
        if not get_firm_by_name(company_name):
            new_firm = {
                "id": str(uuid.uuid4()),
                "name": company_name,
                "location": safe_get(person_data, 'location'),
                "headquarters": "Unknown",
                "aum": "Unknown",
                "founded": None,
                "strategy": safe_get(person_data, 'expertise', 'Hedge Fund'),
                "website": "",
                "description": f"Hedge fund - extracted from newsletter intelligence",
                "performance_metrics": [],
                "created_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                # Historical context for firm
                "extraction_history": [{
                    **source_metadata,
                    "discovered_through": f"Employee extraction: {safe_get(person_data, 'name')}",
                    "original_context": person_data
                }]
            }
            st.session_state.firms.append(new_firm)
        
        # Add employment record with context
        st.session_state.employments.append({
            "id": str(uuid.uuid4()),
            "person_id": new_person_id,
            "company_name": company_name,
            "title": title,
            "start_date": date.today(),
            "end_date": None,
            "location": safe_get(person_data, 'location'),
            "strategy": safe_get(person_data, 'expertise', 'Unknown'),
            "created_date": datetime.now().isoformat(),
            "extraction_context": source_metadata
        })
        
        saved_people += 1
    
    # Process performance metrics with historical context
    for metric in approved_performance:
        fund_name = safe_get(metric, 'fund_name')
        matching_firm = None
        
        for firm in st.session_state.firms:
            firm_name = safe_get(firm, 'name').lower()
            if fund_name.lower() in firm_name or firm_name in fund_name.lower():
                matching_firm = firm
                break
        
        if matching_firm:
            if 'performance_metrics' not in matching_firm:
                matching_firm['performance_metrics'] = []
            
            # Add metric with historical context
            enhanced_metric = {
                **metric,
                "id": str(uuid.uuid4()),
                "recorded_date": datetime.now().isoformat(),
                "extraction_context": source_metadata,
                "source_reliability": "newsletter_extraction"
            }
            
            # Check for duplicates
            existing = any(
                m.get('metric_type') == metric.get('metric_type') and 
                m.get('period') == metric.get('period') and
                m.get('date') == metric.get('date')
                for m in matching_firm['performance_metrics']
            )
            
            if not existing:
                matching_firm['performance_metrics'].append(enhanced_metric)
                
                # Update firm's last_updated
                matching_firm['last_updated'] = datetime.now().isoformat()
                
                # Add to firm's extraction history if it exists
                if 'extraction_history' not in matching_firm:
                    matching_firm['extraction_history'] = []
                
                matching_firm['extraction_history'].append({
                    **source_metadata,
                    "content_type": "performance_metric",
                    "metric_added": {
                        "type": safe_get(metric, 'metric_type'),
                        "value": safe_get(metric, 'value'),
                        "period": safe_get(metric, 'period')
                    }
                })
                
                saved_performance += 1
    
    save_data()
    return saved_people, saved_performance

# --- DATA EXPORT FUNCTIONS ---
def export_to_csv():
    """Quick CSV export of all data"""
    try:
        all_data = []
        
        # Export people
        for person in st.session_state.people:
            all_data.append({
                'Type': 'Person',
                'Name': safe_get(person, 'name'),
                'Title': safe_get(person, 'current_title'),
                'Company': safe_get(person, 'current_company_name'),
                'Location': safe_get(person, 'location'),
                'Email': safe_get(person, 'email'),
                'Expertise': safe_get(person, 'expertise'),
                'AUM': safe_get(person, 'aum_managed')
            })
        
        # Export firms
        for firm in st.session_state.firms:
            all_data.append({
                'Type': 'Firm',
                'Name': safe_get(firm, 'name'),
                'Title': safe_get(firm, 'strategy'),
                'Company': safe_get(firm, 'name'),
                'Location': safe_get(firm, 'location'),
                'Email': safe_get(firm, 'website'),
                'Expertise': safe_get(firm, 'strategy'),
                'AUM': safe_get(firm, 'aum')
            })
        
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return df.to_csv(index=False), f"hedge_fund_data_{timestamp}.csv"
    
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return None, None

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
try:
    initialize_session_state()
except Exception as init_error:
    st.error(f"Initialization error: {init_error}")
    st.stop()

# Check for stuck processing and auto-recover
try:
    recovery_occurred = check_and_recover_stuck_processing()
    if recovery_occurred:
        st.rerun()  # Refresh UI after recovery
except Exception as recovery_error:
    logger.warning(f"Recovery check failed: {recovery_error}")
    # Continue without recovery

# --- HEADER WITH QUICK DOWNLOAD ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("👥 Asian Hedge Fund Talent Network")
    st.markdown("### Professional intelligence platform for Asia's hedge fund industry")

with col2:
    # Background processing widget in header
    display_background_processing_widget()

with col3:
    # Quick CSV download and emergency controls
    col3a, col3b = st.columns(2)
    with col3a:
        csv_data, filename = export_to_csv()
        if csv_data:
            st.download_button(
                "📊 Quick CSV Export",
                csv_data,
                filename,
                "text/csv",
                use_container_width=True,
                help="Download all data as CSV"
            )
    with col3b:
        # Emergency stop button (only show if processing is running)
        if st.session_state.background_processing['is_running']:
            if st.button("🚨 Emergency Stop", use_container_width=True, help="Force stop stuck processing", type="secondary"):
                if emergency_stop_processing():
                    st.warning("🚨 Emergency stop executed!")
                    st.rerun()

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
    
    # Enhanced Model Selection with actual 2.0/2.5 models
    st.markdown("---")
    st.subheader("🤖 Model Selection")
    
    model_options = {
        "Gemini 1.5 Flash (Recommended)": "gemini-1.5-flash",
        "Gemini 1.5 Flash Latest": "gemini-1.5-flash-latest",
        "Gemini 1.5 Flash 8B": "gemini-1.5-flash-8b",
        "Gemini 1.5 Pro": "gemini-1.5-pro",
        "Gemini 1.5 Pro Latest": "gemini-1.5-pro-latest",
        "Gemini 2.0 Flash": "gemini-2.0-flash-thinking-exp",  # Actual 2.0
        "Gemini 2.0 Flash Experimental": "gemini-2.0-flash-exp",
        "Gemini 2.5 Flash": "gemini-2.5-flash-exp",  # Latest 2.5
        "Gemini Experimental 1114": "gemini-exp-1114",
        "Gemini Experimental 1121": "gemini-exp-1121"
    }
    
    selected_model_name = st.selectbox(
        "Choose AI model:",
        options=list(model_options.keys()),
        index=0,
        help="Different models have different capabilities and rate limits"
    )
    
    selected_model_id = model_options[selected_model_name]
    
    # Show rate limits and processing status
    rate_limits = get_model_rate_limits(selected_model_id)
    st.caption(f"⏱️ Rate limit: {rate_limits['requests_per_minute']} req/min, {rate_limits['delay']}s delay")
    
    # Show current processing status if running
    if st.session_state.background_processing['is_running']:
        with st.container(border=True):
            st.warning("🔄 **Processing in Progress**")
            bg_proc = st.session_state.background_processing
            
            # Calculate time properly
            if bg_proc['start_time']:
                elapsed = (datetime.now() - bg_proc['start_time']).total_seconds()
                if elapsed < 60:
                    time_str = f"{int(elapsed)}s"
                elif elapsed < 3600:
                    time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                else:
                    time_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
            else:
                time_str = "Unknown"
            
            st.caption(f"⏱️ Running: {time_str}")
            st.caption(f"📊 Chunk: {bg_proc['current_chunk']}/{bg_proc['total_chunks']}")
            st.caption(f"✅ Found: {bg_proc.get('saved_people', 0) + len(bg_proc['results']['people'])} people")
            st.caption(f"📈 Metrics: {bg_proc.get('saved_performance', 0) + len(bg_proc['results']['performance'])}")
            
            if bg_proc.get('errors'):
                st.caption(f"❌ Errors: {len(bg_proc['errors'])}")
            
            # Check for timeout warning
            if 'last_activity' in bg_proc and bg_proc['last_activity']:
                time_since_activity = (datetime.now() - bg_proc['last_activity']).total_seconds()
                if time_since_activity > 120:  # Warn after 2 minutes
                    st.warning(f"⚠️ No activity for {int(time_since_activity)}s")
            
            col_stop, col_emergency = st.columns(2)
            with col_stop:
                if st.button("⏹️ Stop", key="sidebar_stop", use_container_width=True):
                    st.session_state.background_processing['is_running'] = False
                    st.rerun()
            with col_emergency:
                if st.button("🚨 Emergency", key="sidebar_emergency", use_container_width=True):
                    emergency_stop_processing()
                    st.rerun()
    
    # Processing Configuration
    st.markdown("---")
    st.subheader("⚙️ Processing Configuration")
    
    preprocessing_options = {
        "🚀 Minimal": "minimal",
        "⚖️ Balanced (Recommended)": "balanced", 
        "🔍 Aggressive": "aggressive",
        "📄 None (Raw Content)": "none"
    }
    
    selected_preprocessing = st.selectbox(
        "Text Preprocessing:",
        options=list(preprocessing_options.keys()),
        index=1,
        help="How much filtering to apply to input text"
    )
    preprocessing_mode = preprocessing_options[selected_preprocessing]
    
    chunking_options = {
        "🤖 Auto (Recommended)": "auto",
        "📄 Single Chunk": "single",
        "🔹 Small Chunks (10K)": "small",
        "⚖️ Medium Chunks (20K)": "medium", 
        "🔷 Large Chunks (35K)": "large",
        "🔶 XLarge Chunks (50K)": "xlarge"
    }
    
    selected_chunking = st.selectbox(
        "Chunking Strategy:",
        options=list(chunking_options.keys()),
        index=0,
        help="How to split large files for processing"
    )
    chunk_size_mode = chunking_options[selected_chunking]
    
    # Review Mode Toggle
    st.markdown("---")
    st.subheader("👀 Review Settings")
    
    enable_review = st.checkbox(
        "📋 Enable Review Mode", 
        value=st.session_state.enable_review_mode,
        help="Review extracted data before saving to database"
    )
    st.session_state.enable_review_mode = enable_review
    
    if enable_review:
        timeout_options = {"2 minutes": 120, "3 minutes": 180, "5 minutes": 300, "10 minutes": 600}
        selected_timeout = st.selectbox("⏱️ Auto-save timeout:", options=list(timeout_options.keys()), index=1)
        st.session_state.auto_save_timeout = timeout_options[selected_timeout]
        
        if st.session_state.pending_review_data:
            remaining = get_review_time_remaining()
            if remaining > 0:
                minutes_left = int(remaining // 60)
                seconds_left = int(remaining % 60)
                st.warning(f"⏱️ **{len(st.session_state.pending_review_data)} items pending review!**")
                st.info(f"Auto-save in: {minutes_left}m {seconds_left}s")
                
                if st.button("🔍 Go to Review", use_container_width=True, key="sidebar_goto_review"):
                    st.session_state.show_review_interface = True
                    st.rerun()
    
    # Setup model
    model = None
    if api_key and GENAI_AVAILABLE:
        model = setup_gemini(api_key, selected_model_id)
        
        st.markdown("---")
        st.subheader("📄 Extract from Newsletter")
        
        input_method = st.radio("Input method:", ["📝 Text", "📁 File"])
        
        newsletter_text = ""
        if input_method == "📝 Text":
            newsletter_text = st.text_area("Newsletter content:", height=200, 
                                         placeholder="Paste hedge fund newsletter content here...")
        else:
            uploaded_file = st.file_uploader("Upload newsletter:", type=['txt'])
            if uploaded_file:
                try:
                    # Try enhanced file loading first
                    success, content, error_msg, encoding_used = load_file_content_enhanced(uploaded_file)
                    
                    if success:
                        newsletter_text = content
                        char_count = len(newsletter_text)
                        
                        # Show file info with encoding
                        st.success(f"✅ **File loaded successfully!**")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.info(f"📁 **Size**: {char_count:,} characters")
                            st.info(f"🔤 **Encoding**: {encoding_used}")
                        with col_info2:
                            # Calculate estimates based on current settings
                            if chunk_size_mode == "auto":
                                estimated_chunk_size = min(max(char_count // 50, 15000), 35000)
                            else:
                                chunk_sizes = {"single": 25000, "small": 10000, "medium": 20000, "large": 35000, "xlarge": 50000}
                                estimated_chunk_size = chunk_sizes.get(chunk_size_mode, 20000)
                            
                            estimated_chunks = max(1, char_count // estimated_chunk_size)
                            estimated_time = estimated_chunks * 2  # 2 minutes per chunk estimate
                            
                            st.info(f"📊 **Est. chunks**: {estimated_chunks} ({chunk_size_mode} mode)")
                            st.info(f"⏱️ **Est. time**: ~{estimated_time} minutes")
                        
                        # Show warning message if there was one
                        if error_msg:
                            st.warning(f"⚠️ {error_msg}")
                        
                        # Show preview of content
                        with st.expander("👀 Content Preview", expanded=False):
                            preview_text = newsletter_text[:1000] + "..." if len(newsletter_text) > 1000 else newsletter_text
                            st.text_area("Preview:", value=preview_text, height=150, disabled=True)
                    else:
                        st.error(f"❌ {error_msg}")
                        st.info("💡 **Tips**: Try saving the file as UTF-8 text, or check if it contains special characters")
                        
                except NameError as name_error:
                    # Fallback to simple file loading if enhanced function fails
                    st.warning("Using fallback file loading...")
                    try:
                        newsletter_text = uploaded_file.getvalue().decode('utf-8')
                        st.success(f"✅ File loaded: {len(newsletter_text):,} characters (UTF-8)")
                    except UnicodeDecodeError:
                        try:
                            newsletter_text = uploaded_file.getvalue().decode('cp1252')
                            st.success(f"✅ File loaded: {len(newsletter_text):,} characters (Windows-1252)")
                        except:
                            try:
                                newsletter_text = uploaded_file.getvalue().decode('utf-8', errors='replace')
                                st.warning(f"⚠️ File loaded with character replacements: {len(newsletter_text):,} characters")
                            except Exception as fallback_error:
                                st.error(f"❌ Could not load file: {fallback_error}")
                                newsletter_text = ""
                                
                except Exception as file_error:
                    st.error(f"❌ Error loading file: {str(file_error)}")
                    st.info("💡 **Try**: Different file encoding or copy/paste the content instead")

        # Extract button
        if st.button("🚀 Start Background Extraction", use_container_width=True):
            if not newsletter_text.strip():
                st.error("Please provide newsletter content")
            elif not model:
                st.error("Please provide API key")
            elif st.session_state.background_processing['is_running']:
                st.warning("Extraction already in progress!")
            else:
                # Start background processing
                start_background_extraction(newsletter_text, model, preprocessing_mode, chunk_size_mode)
                st.success("🚀 Background extraction started!")
                st.rerun()

    elif not GENAI_AVAILABLE:
        st.error("Please install: pip install google-generativeai")

# --- MAIN CONTENT AREA ---

# Review Interface (if active)
if st.session_state.show_review_interface and st.session_state.pending_review_data:
    display_review_interface()

# Global Search Bar (SINGLE)
st.markdown("---")
col1, col2 = st.columns([4, 1])

with col1:
    search_query = st.text_input(
        "🔍 Search people, firms, or performance...", 
        value=st.session_state.global_search,
        placeholder="Try: 'Goldman Sachs', 'Portfolio Manager', 'Citadel', 'Sharpe ratio'...",
        key="main_search_input"
    )

with col2:
    if st.button("🔍 Search", use_container_width=True, key="main_search_button") or search_query != st.session_state.global_search:
        st.session_state.global_search = search_query
        if search_query and len(search_query.strip()) >= 2:
            st.rerun()

# Handle global search results
if st.session_state.global_search and len(st.session_state.global_search.strip()) >= 2:
    search_query = st.session_state.global_search
    matching_people, matching_firms, matching_metrics = enhanced_global_search(search_query)
    
    if matching_people or matching_firms or matching_metrics:
        st.markdown("### 🔍 Search Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 People", len(matching_people))
        with col2:
            st.metric("🏢 Firms", len(matching_firms))
        with col3:
            st.metric("📊 Metrics", len(matching_metrics))
        with col4:
            total_results = len(matching_people) + len(matching_firms) + len(matching_metrics)
            st.metric("🎯 Total", total_results)
        
        # Show search results (limited to save space)
        if matching_people:
            st.markdown(f"**👥 People ({len(matching_people)} found)**")
            for person in matching_people[:3]:  # Show first 3
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
        
        if matching_firms:
            st.markdown(f"**🏢 Firms ({len(matching_firms)} found)**")
            for firm in matching_firms[:3]:  # Show first 3
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{safe_get(firm, 'name')}**")
                    st.caption(f"{safe_get(firm, 'strategy')} • {safe_get(firm, 'location')}")
                with col2:
                    st.caption(f"💰 {safe_get(firm, 'aum')}")
                with col3:
                    if st.button("👁️ View", key=f"search_firm_{firm['id']}", use_container_width=True):
                        go_to_firm_details(firm['id'])
                        st.rerun()
        
        # Clear search button
        if st.button("❌ Clear Search", key="main_clear_search"):
            st.session_state.global_search = ""
            st.rerun()
        
        st.markdown("---")
    
    else:
        st.info(f"🔍 No results found for '{search_query}'. Try different keywords.")
        if st.button("❌ Clear Search", key="no_results_clear_search"):
            st.session_state.global_search = ""
            st.rerun()
        st.markdown("---")

# Top Navigation (without Performance tab)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

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
    if st.button("➕ Add Person", use_container_width=True):
        st.session_state.show_add_person_modal = True
        st.rerun()

with col4:
    if st.button("🏢➕ Add Firm", use_container_width=True):
        st.session_state.show_add_firm_modal = True
        st.rerun()

with col5:
    # Quick stats
    col5a, col5b, col5c = st.columns(3)
    with col5a:
        st.metric("People", len(st.session_state.people))
    with col5b:
        st.metric("Firms", len(st.session_state.firms))
    with col5c:
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
            
            # Company selection with existing options
            company_options = [f['name'] for f in st.session_state.firms if f.get('name')]
            if company_options:
                company = st.text_input(
                    "Current Company*",
                    placeholder="Type company name or select from suggestions",
                    help=f"Available: {', '.join(company_options[:3])}{'...' if len(company_options) > 3 else ''}"
                )
            else:
                company = st.text_input("Current Company*", placeholder="Company Name")
            
            location = handle_dynamic_input("location", "", "people", "add_person")
        
        with col2:
            email = st.text_input("Email", placeholder="john.smith@company.com")
            phone = st.text_input("Phone", placeholder="+852-1234-5678")
            education = st.text_input("Education", placeholder="Harvard, MIT")
            expertise = handle_dynamic_input("expertise", "", "people", "add_person")
        
        col3, col4 = st.columns(2)
        with col3:
            aum_managed = st.text_input("AUM Managed", placeholder="500M USD")
        with col4:
            strategy = handle_dynamic_input("strategy", "", "people", "add_person")
        
        submitted = st.form_submit_button("Add Person", use_container_width=True)
        
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
                    "strategy": strategy,
                    "created_date": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "extraction_history": [{
                        "extraction_date": datetime.now().isoformat(),
                        "source_type": "manual_entry",
                        "context_preview": f"Manually added person: {name} at {company}",
                        "entry_method": "add_person_form"
                    }]
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
                    "strategy": strategy or "Unknown",
                    "created_date": datetime.now().isoformat(),
                    "extraction_context": {
                        "extraction_date": datetime.now().isoformat(),
                        "source_type": "manual_entry"
                    }
                })
                
                save_data()
                st.success(f"✅ Added {name}!")
                st.session_state.show_add_person_modal = False
                st.rerun()
            else:
                st.error("Please fill required fields (*)")
    
    if st.button("❌ Cancel", key="cancel_add_person_outside"):
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
            location = handle_dynamic_input("location", "", "firms", "add_firm")
            aum = st.text_input("AUM", placeholder="5B USD")
            
        with col2:
            strategy = handle_dynamic_input("strategy", "", "firms", "add_firm")
            founded = st.number_input("Founded", min_value=1900, max_value=2025, value=2000)
            website = st.text_input("Website", placeholder="https://company.com")
        
        description = st.text_area("Description", placeholder="Brief description of the firm...")
        
        submitted = st.form_submit_button("Add Firm", use_container_width=True)
        
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
                    "description": description if description else f"{strategy} hedge fund based in {location}",
                    "performance_metrics": [],
                    "created_date": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "extraction_history": [{
                        "extraction_date": datetime.now().isoformat(),
                        "source_type": "manual_entry",
                        "context_preview": f"Manually added firm: {firm_name} in {location}",
                        "entry_method": "add_firm_form"
                    }]
                })
                
                save_data()
                st.success(f"✅ Added {firm_name}!")
                st.session_state.show_add_firm_modal = False
                st.rerun()
            else:
                st.error("Please fill Firm Name and Location")
    
    if st.button("❌ Cancel", key="cancel_add_firm_outside"):
        st.session_state.show_add_firm_modal = False
        st.rerun()

# --- PEOPLE VIEW WITH CARD LAYOUT ---
if st.session_state.current_view == 'people':
    st.markdown("---")
    st.header("👥 Hedge Fund Professionals")
    
    if not st.session_state.people:
        st.info("No people added yet. Use 'Add Person' button above or extract from newsletters using AI.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            locations = ["All"] + sorted(list(set(safe_get(p, 'location') for p in st.session_state.people if safe_get(p, 'location') not in ['Unknown', 'N/A', ''])))
            location_filter = st.selectbox("Filter by Location", locations)
        with col2:
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
        
        # Sort by most recent first (using created_date or last_updated if available)
        def get_person_date(person):
            # Try multiple date fields, default to very old date if none found
            for date_field in ['last_updated', 'created_date']:
                if date_field in person and person[date_field]:
                    try:
                        return datetime.fromisoformat(person[date_field].replace('Z', '+00:00'))
                    except:
                        continue
            return datetime(2000, 1, 1)  # Default old date
        
        filtered_people.sort(key=get_person_date, reverse=True)
        
        # Paginate results
        people_to_show, people_page_info = paginate_data(filtered_people, st.session_state.people_page, 12)  # 12 cards per page
        
        st.write(f"**Showing {people_page_info['start_idx'] + 1}-{people_page_info['end_idx']} of {people_page_info['total_items']} people** (most recent first)")
        
        # Display people in card grid (3 columns)
        if people_to_show:
            for i in range(0, len(people_to_show), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(people_to_show):
                        person = people_to_show[i + j]
                        
                        with col:
                            # Create card container
                            with st.container(border=True):
                                # Person header
                                st.markdown(f"**👤 {safe_get(person, 'name')}**")
                                st.caption(f"{safe_get(person, 'current_title')}")
                                st.caption(f"🏢 {safe_get(person, 'current_company_name')}")
                                
                                # Person details
                                location = safe_get(person, 'location')
                                if location != 'Unknown':
                                    st.text(f"📍 {location}")
                                
                                expertise = safe_get(person, 'expertise')
                                if expertise and expertise != 'Unknown':
                                    st.text(f"🏆 {expertise}")
                                
                                aum = safe_get(person, 'aum_managed')
                                if aum and aum != 'Unknown':
                                    st.text(f"💰 {aum}")
                                
                                # Performance metrics count
                                person_metrics = get_person_performance_metrics(person['id'])
                                if person_metrics:
                                    st.text(f"📊 {len(person_metrics)} metrics")
                                
                                # Show when added/updated
                                if 'created_date' in person or 'last_updated' in person:
                                    date_to_show = person.get('last_updated') or person.get('created_date')
                                    if date_to_show:
                                        try:
                                            date_obj = datetime.fromisoformat(date_to_show.replace('Z', '+00:00'))
                                            days_ago = (datetime.now(date_obj.tzinfo) - date_obj).days
                                            if days_ago == 0:
                                                st.caption("🕒 Added today")
                                            elif days_ago == 1:
                                                st.caption("🕒 Added yesterday")
                                            elif days_ago < 7:
                                                st.caption(f"🕒 Added {days_ago} days ago")
                                            else:
                                                st.caption(f"🕒 Added {date_obj.strftime('%b %d, %Y')}")
                                        except:
                                            pass
                                
                                # Action buttons
                                col_view, col_edit = st.columns(2)
                                with col_view:
                                    if st.button("👁️ View", key=f"view_person_card_{person['id']}", use_container_width=True):
                                        go_to_person_details(person['id'])
                                        st.rerun()
                                with col_edit:
                                    if st.button("✏️ Edit", key=f"edit_person_card_{person['id']}", use_container_width=True):
                                        st.session_state.edit_person_data = person
                                        st.session_state.show_edit_person_modal = True
                                        st.rerun()
        
        # Pagination controls
        display_pagination_controls(people_page_info, "people")

# --- FIRMS VIEW WITH CARD LAYOUT ---
elif st.session_state.current_view == 'firms':
    st.markdown("---")
    st.header("🏢 Hedge Funds in Asia")
    
    if not st.session_state.firms:
        st.info("No firms added yet. Use 'Add Firm' button above.")
    else:
        # Sort firms by most recent first
        def get_firm_date(firm):
            for date_field in ['last_updated', 'created_date']:
                if date_field in firm and firm[date_field]:
                    try:
                        return datetime.fromisoformat(firm[date_field].replace('Z', '+00:00'))
                    except:
                        continue
            return datetime(2000, 1, 1)  # Default old date
        
        sorted_firms = sorted(st.session_state.firms, key=get_firm_date, reverse=True)
        
        # Paginate results
        firms_to_show, firms_page_info = paginate_data(sorted_firms, st.session_state.firms_page, 12)  # 12 cards per page
        
        st.write(f"**Showing {firms_page_info['start_idx'] + 1}-{firms_page_info['end_idx']} of {firms_page_info['total_items']} firms** (most recent first)")
        
        # Display firms in card grid (3 columns)
        if firms_to_show:
            for i in range(0, len(firms_to_show), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(firms_to_show):
                        firm = firms_to_show[i + j]
                        people_count = len(get_people_by_firm(safe_get(firm, 'name')))
                        metrics_count = len(firm.get('performance_metrics', []))
                        
                        with col:
                            # Create card container
                            with st.container(border=True):
                                # Firm header
                                st.markdown(f"**🏢 {safe_get(firm, 'name')}**")
                                st.caption(f"{safe_get(firm, 'strategy')}")
                                
                                # Firm details
                                location = safe_get(firm, 'location')
                                if location != 'Unknown':
                                    st.text(f"📍 {location}")
                                
                                aum = safe_get(firm, 'aum')
                                if aum and aum != 'Unknown':
                                    st.text(f"💰 {aum}")
                                
                                founded = safe_get(firm, 'founded')
                                if founded and founded != 'Unknown':
                                    st.text(f"📅 Founded {founded}")
                                
                                # Metrics
                                col_people, col_metrics = st.columns(2)
                                with col_people:
                                    st.metric("👥", people_count, label_visibility="collapsed", help="People")
                                with col_metrics:
                                    st.metric("📊", metrics_count, label_visibility="collapsed", help="Performance Metrics")
                                
                                # Website
                                website = safe_get(firm, 'website')
                                if website and website != 'Unknown':
                                    st.text(f"🌐 Website")
                                
                                # Show when added/updated
                                if 'created_date' in firm or 'last_updated' in firm:
                                    date_to_show = firm.get('last_updated') or firm.get('created_date')
                                    if date_to_show:
                                        try:
                                            date_obj = datetime.fromisoformat(date_to_show.replace('Z', '+00:00'))
                                            days_ago = (datetime.now(date_obj.tzinfo) - date_obj).days
                                            if days_ago == 0:
                                                st.caption("🕒 Added today")
                                            elif days_ago == 1:
                                                st.caption("🕒 Added yesterday")
                                            elif days_ago < 7:
                                                st.caption(f"🕒 Added {days_ago} days ago")
                                            else:
                                                st.caption(f"🕒 Added {date_obj.strftime('%b %d, %Y')}")
                                        except:
                                            pass
                                
                                # Action buttons
                                col_view, col_edit = st.columns(2)
                                with col_view:
                                    if st.button("👁️ View", key=f"view_firm_card_{firm['id']}", use_container_width=True):
                                        go_to_firm_details(firm['id'])
                                        st.rerun()
                                with col_edit:
                                    if st.button("✏️ Edit", key=f"edit_firm_card_{firm['id']}", use_container_width=True):
                                        st.session_state.edit_firm_data = firm
                                        st.session_state.show_edit_firm_modal = True
                                        st.rerun()
        
        # Pagination controls
        display_pagination_controls(firms_page_info, "firms")

# --- FIRM DETAILS VIEW (with integrated performance) ---
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
    
    # Firm metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Assets Under Management", safe_get(firm, 'aum'))
    with col2:
        st.metric("Founded", safe_get(firm, 'founded'))
    with col3:
        people_count = len(get_people_by_firm(safe_get(firm, 'name')))
        st.metric("Total People", people_count)
    
    # Firm details
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
    
    # INTEGRATED PERFORMANCE METRICS
    st.markdown("---")
    st.subheader("📊 Performance Metrics")
    
    firm_metrics = firm.get('performance_metrics', [])
    if firm_metrics:
        st.write(f"**Found {len(firm_metrics)} performance metrics:**")
        
        # Create performance dataframe for better display
        metrics_data = []
        for metric in firm_metrics:
            metrics_data.append({
                "Metric": safe_get(metric, 'metric_type').title(),
                "Value": safe_get(metric, 'value'),
                "Period": safe_get(metric, 'period'),
                "Date": safe_get(metric, 'date'),
                "Details": safe_get(metric, 'additional_info')
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
    else:
        st.info("No performance metrics found for this firm.")
        st.write("💡 Performance metrics will appear here when extracted from newsletters.")
    
    # People at this firm
    st.markdown("---")
    st.subheader(f"👥 People at {safe_get(firm, 'name')}")
    
    firm_people = get_people_by_firm(safe_get(firm, 'name'))
    if firm_people:
        for person in firm_people[:10]:  # Show first 10
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**👤 {safe_get(person, 'name')}**")
                    st.caption(safe_get(person, 'current_title'))
                
                with col2:
                    email = safe_get(person, 'email')
                    expertise = safe_get(person, 'expertise')
                    if email:
                        st.caption(f"📧 {email}")
                    if expertise:
                        st.caption(f"🏆 {expertise}")
                
                with col3:
                    if st.button("👁️ Profile", key=f"view_full_{person['id']}", use_container_width=True):
                        go_to_person_details(person['id'])
                        st.rerun()
                
                st.markdown("---")
        
        if len(firm_people) > 10:
            st.info(f"Showing first 10 of {len(firm_people)} people")
    else:
        st.info("No people added for this firm yet.")

# --- PERSON DETAILS VIEW (with integrated performance) ---
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
    
    # INTEGRATED PERFORMANCE METRICS
    st.markdown("---")
    st.subheader("📊 Performance Track Record")
    
    person_metrics = get_person_performance_metrics(person['id'])
    if person_metrics:
        st.write(f"**Found {len(person_metrics)} performance metrics:**")
        
        # Create performance dataframe
        metrics_data = []
        for metric in person_metrics:
            metrics_data.append({
                "Metric": safe_get(metric, 'metric_type').title(),
                "Value": safe_get(metric, 'value'),
                "Period": safe_get(metric, 'period'),
                "Date": safe_get(metric, 'date'),
                "Details": safe_get(metric, 'additional_info')
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
    
    else:
        st.info("No performance metrics found for this person.")
        st.write("💡 Performance metrics will appear here when extracted from newsletters.")
    
    # Employment History
    st.markdown("---")
    st.subheader("💼 Employment History")
    
    employments = get_employments_by_person_id(person['id'])
    if employments:
        sorted_employments = sorted(
            [emp for emp in employments if emp.get('start_date')], 
            key=lambda x: x['start_date'], 
            reverse=True
        )
        
        for emp in sorted_employments:
            end_date_str = emp['end_date'].strftime("%B %Y") if emp.get('end_date') else "Present"
            start_date_str = emp['start_date'].strftime("%B %Y") if emp.get('start_date') else "Unknown"
            
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
    
    # Shared Work History
    st.markdown("---")
    st.subheader("🤝 Professional Network Connections")
    
    shared_history = get_shared_work_history(person['id'])
    
    if shared_history:
        st.write(f"**Found {len(shared_history)} colleagues who worked at the same companies:**")
        
        for connection in shared_history[:10]:  # Show first 10
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
            st.info(f"Showing top 10 of {len(shared_history)} connections")
        
    else:
        st.info("No shared work history found with other people in the database.")

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
            company_options = [f['name'] for f in st.session_state.firms if f.get('name')]
            
            company = st.text_input(
                "Company Name*",
                value=current_company,
                placeholder="Type company name",
                help=f"Available firms: {', '.join(company_options[:3])}{'...' if len(company_options) > 3 else ''}"
            )
            
            location = handle_dynamic_input("location", safe_get(person_data, 'location'), "people", "edit_person")
        
        with col2:
            email = st.text_input("Email", value=safe_get(person_data, 'email'))
            phone = st.text_input("Phone", value=safe_get(person_data, 'phone'))
            linkedin = st.text_input("LinkedIn URL", value=safe_get(person_data, 'linkedin_profile_url'))
            education = st.text_input("Education", value=safe_get(person_data, 'education'))
        
        col3, col4 = st.columns(2)
        with col3:
            expertise = handle_dynamic_input("expertise", safe_get(person_data, 'expertise'), "people", "edit_person")
            aum = st.text_input("AUM Managed", value=safe_get(person_data, 'aum_managed'))
        
        with col4:
            strategy = handle_dynamic_input("strategy", safe_get(person_data, 'strategy'), "people", "edit_person")
        
        col_save, col_cancel, col_delete = st.columns(3)
        
        with col_save:
            save_submitted = st.form_submit_button("💾 Save Changes", use_container_width=True)
        with col_cancel:
            cancel_submitted = st.form_submit_button("❌ Cancel", use_container_width=True)
        with col_delete:
            delete_submitted = st.form_submit_button("🗑️ Delete Person", use_container_width=True)
        
        # Handle form submissions
        if save_submitted:
                if name and title and company and location:
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
            location = handle_dynamic_input("location", safe_get(firm_data, 'location'), "firms", "edit_firm")
            headquarters = st.text_input("Headquarters", value=safe_get(firm_data, 'headquarters'))
            aum = st.text_input("AUM", value=safe_get(firm_data, 'aum'))
            
        with col2:
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
                with st.expander(f"{safe_get(metric, 'metric_type')} - {safe_get(metric, 'period')}"):
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.write(f"**Value**: {safe_get(metric, 'value')}")
                        st.write(f"**Period**: {safe_get(metric, 'period')}")
                    with col_metric2:
                        st.write(f"**Date**: {safe_get(metric, 'date')}")
                        st.write(f"**Info**: {safe_get(metric, 'additional_info')}")
                    
                    if st.button(f"🗑️ Remove Metric", key=f"remove_metric_{i}"):
                        existing_metrics.pop(i)
                        firm_data['performance_metrics'] = existing_metrics
                        st.rerun()
        else:
            st.info("No performance metrics yet. Metrics will be added automatically when extracted from newsletters.")
        
        col_save, col_cancel, col_delete = st.columns(3)
        
        with col_save:
            save_submitted = st.form_submit_button("💾 Save Changes", use_container_width=True)
        with col_cancel:
            cancel_submitted = st.form_submit_button("❌ Cancel", use_container_width=True)
        with col_delete:
            delete_submitted = st.form_submit_button("🗑️ Delete Firm", use_container_width=True)
        
        # Handle form submissions
        if save_submitted:
                if firm_name and location:
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

# --- DATA EXPORT & BACKUP SECTION ---
st.markdown("---")
st.markdown("### 📊 Data Export & Backup")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📁 Database Exports")
    
    export_format_options = ["📄 CSV Files (.zip)", "🗄️ JSON Backup"]
    if EXCEL_AVAILABLE:
        export_format_options.insert(0, "📊 Excel (.xlsx)")
    
    export_format = st.radio(
        "Choose export format:",
        export_format_options,
        horizontal=True
    )
    
    # Data selection checkboxes
    st.markdown("**Select data to export:**")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        export_people = st.checkbox("👥 People", value=True, help=f"{len(st.session_state.people)} records")
        export_firms = st.checkbox("🏢 Firms", value=True, help=f"{len(st.session_state.firms)} records")
    with col_b:
        export_employments = st.checkbox("💼 Employment History", value=True, help=f"{len(st.session_state.employments)} records")
        total_metrics = sum(len(f.get('performance_metrics', [])) for f in st.session_state.firms)
        export_performance = st.checkbox("📊 Performance Metrics", value=True, help=f"{total_metrics} records")
    with col_c:
        export_extractions = st.checkbox("🤖 Extraction History", value=False, help=f"{len(st.session_state.all_extractions)} records")
        export_reviews = st.checkbox("📋 Pending Reviews", value=False, help=f"{len(st.session_state.pending_review_data)} batches")

with col2:
    st.markdown("#### 📈 Export Preview")
    
    total_records = 0
    export_details = []
    
    if export_people and st.session_state.people:
        total_records += len(st.session_state.people)
        export_details.append(f"👥 {len(st.session_state.people)} People")
    
    if export_firms and st.session_state.firms:
        total_records += len(st.session_state.firms)
        export_details.append(f"🏢 {len(st.session_state.firms)} Firms")
    
    if export_employments and st.session_state.employments:
        total_records += len(st.session_state.employments)
        export_details.append(f"💼 {len(st.session_state.employments)} Employment Records")
    
    if export_performance:
        perf_count = sum(len(f.get('performance_metrics', [])) for f in st.session_state.firms)
        if perf_count > 0:
            total_records += perf_count
            export_details.append(f"📊 {perf_count} Performance Metrics")
    
    if export_details:
        st.success(f"**Ready to export {total_records} total records:**")
        for detail in export_details:
            st.write(f"• {detail}")
    else:
        st.warning("No data selected for export")

# Export buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Selected Data", use_container_width=True, disabled=total_records == 0):
        try:
            # Create export based on selections
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if export_format == "🗄️ JSON Backup":
                backup_data = {}
                
                if export_people:
                    backup_data["people"] = st.session_state.people
                if export_firms:
                    backup_data["firms"] = st.session_state.firms
                if export_employments:
                    backup_data["employments"] = st.session_state.employments
                if export_extractions:
                    backup_data["extractions"] = st.session_state.all_extractions
                if export_reviews:
                    backup_data["pending_reviews"] = st.session_state.pending_review_data
                
                backup_data.update({
                    "export_timestamp": datetime.now().isoformat(),
                    "total_records": total_records
                })
                
                export_json = json.dumps(backup_data, indent=2, default=str)
                st.download_button(
                    "💾 Download JSON Backup",
                    export_json,
                    f"hedge_fund_backup_{timestamp}.json",
                    "application/json",
                    use_container_width=True
                )
                st.success(f"✅ JSON backup ready! {total_records} records included.")
            
            else:
                # CSV/Excel export
                csv_data, filename = export_to_csv()
                if csv_data:
                    st.download_button(
                        "💾 Download CSV Export",
                        csv_data,
                        filename,
                        "text/csv",
                        use_container_width=True
                    )
                    st.success("✅ CSV export ready!")
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")

with col2:
    if st.button("📄 Export All (CSV)", use_container_width=True):
        csv_data, filename = export_to_csv()
        if csv_data:
            st.download_button(
                "💾 Download Complete CSV",
                csv_data,
                filename,
                "text/csv",
                use_container_width=True
            )
            st.success("✅ Complete CSV export ready!")

with col3:
    if st.button("🗄️ Full JSON Backup", use_container_width=True):
        backup_data = {
            "people": st.session_state.people,
            "firms": st.session_state.firms,
            "employments": st.session_state.employments,
            "extractions": st.session_state.all_extractions,
            "pending_reviews": st.session_state.pending_review_data,
            "export_timestamp": datetime.now().isoformat(),
            "total_records": len(st.session_state.people) + len(st.session_state.firms) + len(st.session_state.employments)
        }
        
        export_json = json.dumps(backup_data, indent=2, default=str)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        st.download_button(
            "💾 Download Full Backup",
            export_json,
            f"hedge_fund_full_backup_{timestamp}.json",
            "application/json",
            use_container_width=True
        )
        st.success("✅ Full backup ready!")

# --- FOOTER ---
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
    st.markdown("**📋 Smart Review System**")

# Auto-save functionality
current_time = datetime.now()
if 'last_auto_save' not in st.session_state:
    st.session_state.last_auto_save = current_time

time_since_save = (current_time - st.session_state.last_auto_save).total_seconds()
if time_since_save > 30 and (st.session_state.people or st.session_state.firms):
    save_data()
    st.session_state.last_auto_save = current_time

# Handle review timeout
if st.session_state.pending_review_data and st.session_state.review_start_time:
    if get_review_time_remaining() <= 0:
        # Auto-save logic for reviews
        for review_item in st.session_state.pending_review_data:
            approved_people, approved_performance = approve_all_in_review(review_item['id'])
            save_approved_extractions(approved_people, approved_performance)
        
        st.session_state.pending_review_data = []
        st.session_state.show_review_interface = False
