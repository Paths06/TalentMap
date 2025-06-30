import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime, date
import time
import os
from pathlib import Path

# Try to import google.generativeai, handle if not available
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.error("google-generativeai package not installed. Please run: pip install google-generativeai")

# Configure page
st.set_page_config(
    page_title="Hedge Fund Talent Intelligence",
    page_icon="🎯",
    layout="wide"
)

# Database setup
DATA_DIR = Path("talent_data")
DATA_DIR.mkdir(exist_ok=True)

EXTRACTIONS_FILE = DATA_DIR / "extractions.json"
PEOPLE_FILE = DATA_DIR / "people.json"
FIRMS_FILE = DATA_DIR / "firms.json"

# Persistent data functions
def load_data():
    """Load data from JSON files"""
    try:
        extractions = []
        people = []
        firms = []
        
        if EXTRACTIONS_FILE.exists():
            with open(EXTRACTIONS_FILE, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
        
        if PEOPLE_FILE.exists():
            with open(PEOPLE_FILE, 'r', encoding='utf-8') as f:
                people = json.load(f)
                
        if FIRMS_FILE.exists():
            with open(FIRMS_FILE, 'r', encoding='utf-8') as f:
                firms = json.load(f)
                
        return extractions, people, firms
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], [], []

def save_data(extractions=None, people=None, firms=None):
    """Save data to JSON files"""
    try:
        if extractions is not None:
            with open(EXTRACTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(extractions, f, indent=2, default=str)
        
        if people is not None:
            with open(PEOPLE_FILE, 'w', encoding='utf-8') as f:
                json.dump(people, f, indent=2, default=str)
        
        if firms is not None:
            with open(FIRMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(firms, f, indent=2, default=str)
        
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

def auto_save():
    """Auto-save current session state"""
    if 'extractions' in st.session_state and 'people' in st.session_state and 'firms' in st.session_state:
        return save_data(
            extractions=st.session_state.extractions,
            people=st.session_state.people,
            firms=st.session_state.firms
        )
    return False

# Initialize session state
def init_session_state():
    extractions, people, firms = load_data()
    
    if 'extractions' not in st.session_state:
        st.session_state.extractions = extractions
    if 'people' not in st.session_state:
        st.session_state.people = people
    if 'firms' not in st.session_state:
        st.session_state.firms = firms
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'extractions'
    if 'edit_person_id' not in st.session_state:
        st.session_state.edit_person_id = None

# AI setup
@st.cache_resource
def setup_gemini_safe(api_key):
    """Setup Gemini AI model safely"""
    if not GENAI_AVAILABLE:
        st.error("Google Generative AI package not available")
        return None
    
    if not api_key:
        st.error("API key is required")
        return None
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')  # Using stable model name
        return model
    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None

# File reading
def read_file_safely(uploaded_file, max_size_kb=200):
    """Safely read uploaded file with size and encoding checks"""
    try:
        if uploaded_file is None:
            return None
            
        file_size = len(uploaded_file.getvalue())
        if file_size > max_size_kb * 1024:
            st.error(f"File too large: {file_size/1024:.1f}KB. Max: {max_size_kb}KB")
            return None
            
        raw_data = uploaded_file.getvalue()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                return raw_data.decode(encoding)
            except UnicodeDecodeError:
                continue
                
        st.error("Could not decode file with any supported encoding")
        return None
    except Exception as e:
        st.error(f"File reading error: {e}")
        return None

# AI extraction functions
def extract_simple(text, model):
    """Extract people and career movements from text using AI"""
    if not model:
        st.error("AI model not available")
        return []
        
    try:
        if len(text) > 15000:
            text = text[:15000]
            st.warning("Text truncated to 15K characters for processing")
            
        prompt = f"""Extract people and career movements from this financial newsletter. Return JSON only, no other text:

{text}

{{"people": [{{"name": "Full Name", "company": "Company", "role": "Position", "type": "hire/promotion/launch"}}]}}

Find ALL people mentioned in professional contexts like hires, promotions, launches, moves."""
        
        response = model.generate_content(prompt)
        if not response or not response.text:
            st.warning("No response from AI model")
            return []
            
        response_text = response.text.strip()
        
        # Find JSON in response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            st.warning("No valid JSON found in AI response")
            return []
            
        json_text = response_text[json_start:json_end]
        result = json.loads(json_text)
        
        people = result.get('people', [])
        if not people:
            st.info("No people found in the text")
            
        return people
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        st.error(f"Extraction error: {e}")
        return []

def extract_chunked_safe(text, model, chunk_size=12000, max_chunks=5):
    """Extract from large text by processing in chunks"""
    if not model:
        st.error("AI model not available")
        return []
        
    try:
        all_extractions = []
        processed_chars = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        chunk_num = 0
        while processed_chars < len(text) and chunk_num < max_chunks:
            chunk_num += 1
            
            start_pos = max(0, processed_chars - 500)  # Overlap for context
            end_pos = min(start_pos + chunk_size, len(text))
            chunk_text = text[start_pos:end_pos]
            
            status_text.info(f"Processing chunk {chunk_num}/{min(max_chunks, (len(text)//chunk_size) + 1)}...")
            
            if chunk_num > 1:
                time.sleep(3)  # Rate limiting
            
            try:
                chunk_extractions = extract_simple(chunk_text, model)
                
                if chunk_extractions:
                    # Deduplicate by name (case insensitive)
                    existing_names = {ext.get('name', '').lower().strip() 
                                    for ext in all_extractions if ext.get('name')}
                    
                    new_extractions = [ext for ext in chunk_extractions 
                                     if (ext.get('name', '').lower().strip() not in existing_names 
                                         and ext.get('name', '').strip())]
                    
                    all_extractions.extend(new_extractions)
                
            except Exception as e:
                st.warning(f"Error processing chunk {chunk_num}: {e}")
            
            processed_chars = end_pos
            progress_bar.progress(min(processed_chars / len(text), 1.0))
        
        progress_bar.progress(1.0)
        status_text.success(f"Completed: {len(all_extractions)} unique people found")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return all_extractions
        
    except Exception as e:
        st.error(f"Chunked extraction error: {e}")
        return []

# Initialize session state
init_session_state()

# Main header
st.title("🎯 Hedge Fund Talent Intelligence")

# Navigation
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🤖 AI Extraction", use_container_width=True, 
                 type="primary" if st.session_state.current_view == 'extractions' else "secondary"):
        st.session_state.current_view = 'extractions'
        st.rerun()

with col2:
    if st.button("👥 People Database", use_container_width=True,
                 type="primary" if st.session_state.current_view == 'people' else "secondary"):
        st.session_state.current_view = 'people'
        st.session_state.edit_person_id = None
        st.rerun()

with col3:
    if st.button("📊 Analytics", use_container_width=True,
                 type="primary" if st.session_state.current_view == 'analytics' else "secondary"):
        st.session_state.current_view = 'analytics'
        st.rerun()

with col4:
    if st.button("⚙️ Settings", use_container_width=True,
                 type="primary" if st.session_state.current_view == 'settings' else "secondary"):
        st.session_state.current_view = 'settings'
        st.rerun()

st.markdown("---")

# VIEW: AI Extraction
if st.session_state.current_view == 'extractions':
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Newsletter")
        
        # API Key
        api_key = None
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except:
            pass
            
        if not api_key:
            api_key = st.text_input("Gemini API Key:", type="password", 
                                  help="Get your API key from https://makersuite.google.com/app/apikey")
        
        model = setup_gemini_safe(api_key) if api_key else None
        
        # Processing mode
        processing_mode = st.selectbox(
            "Processing mode:",
            ["🛡️ Safe (15K chars)", "⚡ Chunked (Full file)"],
            help="Safe: Quick processing, truncates long files. Chunked: Complete file analysis with rate limiting."
        )
        
        # File upload
        uploaded_file = st.file_uploader("Choose newsletter file:", type=['txt', 'md'])
        
        newsletter_content = None
        if uploaded_file:
            newsletter_content = read_file_safely(uploaded_file)
            if newsletter_content:
                st.success(f"File loaded: {len(newsletter_content):,} characters")
            else:
                st.error("Failed to read file")
        
        # Manual input
        if not newsletter_content:
            newsletter_content = st.text_area("Or paste newsletter text:", height=200, 
                                            placeholder="Paste your newsletter content here...")
        
        # Extract button
        extraction_disabled = not (newsletter_content and model and newsletter_content.strip())
        
        if st.button("🚀 Extract Talent", use_container_width=True, disabled=extraction_disabled):
            if not GENAI_AVAILABLE:
                st.error("Please install google-generativeai: pip install google-generativeai")
            elif not newsletter_content.strip():
                st.error("Please provide newsletter content")
            elif not model:
                st.error("Please provide a valid API key")
            else:
                with st.spinner("Processing newsletter..."):
                    start_time = time.time()
                    
                    if processing_mode.startswith("🛡️"):
                        extractions = extract_simple(newsletter_content, model)
                    else:
                        extractions = extract_chunked_safe(newsletter_content, model)
                    
                    if extractions:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for ext in extractions:
                            ext.update({
                                'timestamp': timestamp,
                                'mode': processing_mode.split()[0],
                                'id': str(uuid.uuid4())
                            })
                        
                        st.session_state.extractions.extend(extractions)
                        auto_save()
                        
                        elapsed = time.time() - start_time
                        st.success(f"✅ Found {len(extractions)} people in {elapsed:.1f}s")
                        st.rerun()
                    else:
                        st.warning("⚠️ No people found in the newsletter")
    
    with col2:
        st.subheader("📊 Recent Extractions")
        
        if st.session_state.extractions:
            # Show recent extractions
            recent = st.session_state.extractions[-10:]
            
            for idx, ext in enumerate(reversed(recent)):
                # Ensure each extraction has a unique ID
                ext_id = ext.get('id', f"ext_{len(st.session_state.extractions) - idx}")
                
                with st.expander(f"{ext.get('name', 'Unknown')} → {ext.get('company', 'Unknown')}"):
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    
                    with col_a:
                        st.write(f"**Role:** {ext.get('role', 'Unknown')}")
                        st.write(f"**Type:** {ext.get('type', 'Unknown')}")
                    
                    with col_b:
                        st.write(f"**Date:** {ext.get('timestamp', '')[:10]}")
                        st.write(f"**Mode:** {ext.get('mode', 'Unknown')}")
                    
                    with col_c:
                        # Check if already in people DB
                        person_exists = any(p.get('name', '').lower() == ext.get('name', '').lower() 
                                          for p in st.session_state.people)
                        
                        if not person_exists:
                            # Use a guaranteed unique key
                            button_key = f"add_{ext_id}_{idx}"
                            if st.button("➕ Add", key=button_key):
                                new_person = {
                                    "id": str(uuid.uuid4()),
                                    "name": ext.get('name', 'Unknown'),
                                    "current_title": ext.get('role', 'Unknown'),
                                    "current_company_name": ext.get('company', 'Unknown'),
                                    "location": "",
                                    "email": "",
                                    "phone": "",
                                    "education": "",
                                    "expertise": "",
                                    "aum_managed": "",
                                    "source_extraction_id": ext_id
                                }
                                st.session_state.people.append(new_person)
                                auto_save()
                                st.rerun()
                        else:
                            st.success("✓ Added")
        else:
            st.info("No extractions yet. Upload a newsletter to start.")

# VIEW: People Database
elif st.session_state.current_view == 'people':
    
    if st.session_state.edit_person_id:
        # Edit mode
        person = next((p for p in st.session_state.people if p['id'] == st.session_state.edit_person_id), None)
        
        if person:
            st.subheader(f"✏️ Edit {person.get('name', 'Person')}")
            
            with st.form("edit_person_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Full Name", value=person.get('name', ''))
                    title = st.text_input("Current Title", value=person.get('current_title', ''))
                    company = st.text_input("Current Company", value=person.get('current_company_name', ''))
                    location = st.text_input("Location", value=person.get('location', ''))
                    email = st.text_input("Email", value=person.get('email', ''))
                
                with col2:
                    phone = st.text_input("Phone", value=person.get('phone', ''))
                    education = st.text_input("Education", value=person.get('education', ''))
                    expertise = st.text_input("Expertise", value=person.get('expertise', ''))
                    aum_managed = st.text_input("AUM Managed", value=person.get('aum_managed', ''))
                
                col_save, col_cancel, col_delete = st.columns([1, 1, 1])
                
                with col_save:
                    if st.form_submit_button("💾 Save Changes", use_container_width=True):
                        # Update person
                        person.update({
                            'name': name,
                            'current_title': title,
                            'current_company_name': company,
                            'location': location,
                            'email': email,
                            'phone': phone,
                            'education': education,
                            'expertise': expertise,
                            'aum_managed': aum_managed
                        })
                        auto_save()
                        st.session_state.edit_person_id = None
                        st.success("Changes saved!")
                        st.rerun()
                
                with col_cancel:
                    if st.form_submit_button("❌ Cancel", use_container_width=True):
                        st.session_state.edit_person_id = None
                        st.rerun()
                
                with col_delete:
                    if st.form_submit_button("🗑️ Delete", use_container_width=True):
                        st.session_state.people = [p for p in st.session_state.people if p['id'] != person['id']]
                        auto_save()
                        st.session_state.edit_person_id = None
                        st.success("Person deleted!")
                        st.rerun()
    
    else:
        # List mode
        st.subheader("👥 People Database")
        
        if st.session_state.people:
            
            # Search and filter
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_term = st.text_input("🔍 Search people...", placeholder="Name, company, title...")
            with col2:
                show_count = st.selectbox("Show", [10, 25, 50, 100], index=1)
            with col3:
                if st.button("📥 Export CSV", use_container_width=True):
                    df = pd.DataFrame(st.session_state.people)
                    csv = df.to_csv(index=False)
                    st.download_button("Download", csv, "people.csv", "text/csv", use_container_width=True)
            
            # Filter people
            filtered_people = st.session_state.people
            if search_term:
                search_lower = search_term.lower()
                filtered_people = [
                    p for p in st.session_state.people
                    if search_lower in p.get('name', '').lower() or
                       search_lower in p.get('current_company_name', '').lower() or
                       search_lower in p.get('current_title', '').lower()
                ]
            
            # Display people in cards
            people_to_show = filtered_people[:show_count]
            
            for person in people_to_show:
                with st.container():
                    col_info, col_actions = st.columns([4, 1])
                    
                    with col_info:
                        st.write(f"**{person.get('name', 'Unknown')}**")
                        st.write(f"{person.get('current_title', 'Unknown')} at {person.get('current_company_name', 'Unknown')}")
                        
                        details = []
                        if person.get('location'):
                            details.append(f"📍 {person['location']}")
                        if person.get('email'):
                            details.append(f"📧 {person['email']}")
                        if person.get('aum_managed'):
                            details.append(f"💰 {person['aum_managed']}")
                        
                        if details:
                            st.caption(" • ".join(details))
                    
                    with col_actions:
                        if st.button("✏️ Edit", key=f"edit_{person['id']}"):
                            st.session_state.edit_person_id = person['id']
                            st.rerun()
                    
                    st.markdown("---")
            
            if len(filtered_people) > show_count:
                st.info(f"Showing {show_count} of {len(filtered_people)} people")
        
        else:
            st.info("No people in database. Add some from AI extractions first.")

# VIEW: Analytics
elif st.session_state.current_view == 'analytics':
    st.subheader("📊 Analytics")
    
    if st.session_state.extractions and st.session_state.people:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Extractions", len(st.session_state.extractions))
        
        with col2:
            st.metric("People in Database", len(st.session_state.people))
        
        with col3:
            companies = set(ext.get('company', '') for ext in st.session_state.extractions if ext.get('company'))
            st.metric("Unique Companies", len(companies))
        
        with col4:
            conversion_rate = (len(st.session_state.people) / len(st.session_state.extractions) * 100) if st.session_state.extractions else 0
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        # Top companies
        st.subheader("🏢 Top Companies")
        company_counts = {}
        for ext in st.session_state.extractions:
            company = ext.get('company', '')
            if company and company != 'Unknown':
                company_counts[company] = company_counts.get(company, 0) + 1
        
        if company_counts:
            top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            company_df = pd.DataFrame(top_companies, columns=['Company', 'Mentions'])
            st.dataframe(company_df, use_container_width=True, hide_index=True)
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        recent_extractions = st.session_state.extractions[-10:]
        
        activity_data = []
        for ext in recent_extractions:
            activity_data.append({
                'Date': ext.get('timestamp', '')[:10],
                'Person': ext.get('name', 'Unknown'),
                'Company': ext.get('company', 'Unknown'),
                'Type': ext.get('type', 'Unknown')
            })
        
        if activity_data:
            activity_df = pd.DataFrame(activity_data)
            st.dataframe(activity_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("No data available for analytics yet.")

# VIEW: Settings
elif st.session_state.current_view == 'settings':
    st.subheader("⚙️ Settings & Data Management")
    
    # Database status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💎 Extractions", len(st.session_state.extractions))
    with col2:
        st.metric("👥 People", len(st.session_state.people))
    with col3:
        st.metric("🏢 Firms", len(st.session_state.firms))
    
    st.markdown("---")
    
    # Data operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Export & Backup")
        
        if st.button("📥 Export All Data", use_container_width=True):
            export_data = {
                "extractions": st.session_state.extractions,
                "people": st.session_state.people,
                "firms": st.session_state.firms,
                "export_timestamp": datetime.now().isoformat()
            }
            
            export_json = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "Download Database",
                export_json,
                f"talent_db_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json",
                use_container_width=True
            )
    
    with col2:
        st.subheader("🧹 Data Management")
        
        if st.button("🧹 Remove Duplicates", use_container_width=True):
            # Remove duplicate extractions
            original_count = len(st.session_state.extractions)
            seen = set()
            unique_extractions = []
            
            for ext in st.session_state.extractions:
                key = f"{ext.get('name', '').lower()}|{ext.get('company', '').lower()}"
                if key not in seen and ext.get('name') and ext.get('company'):
                    seen.add(key)
                    unique_extractions.append(ext)
            
            st.session_state.extractions = unique_extractions
            auto_save()
            
            removed = original_count - len(unique_extractions)
            st.success(f"Removed {removed} duplicate extractions")
            st.rerun()
        
        if st.button("🔄 Force Save", use_container_width=True):
            if auto_save():
                st.success("Data saved successfully!")
        
        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            st.warning("These actions cannot be undone!")
            
            if st.button("🗑️ Clear All Extractions"):
                st.session_state.extractions = []
                auto_save()
                st.success("All extractions cleared")
                st.rerun()
            
            if st.button("🗑️ Clear All People"):
                st.session_state.people = []
                auto_save()
                st.success("All people cleared")
                st.rerun()

# Footer
st.markdown("---")
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("🎯 Hedge Fund Talent Intelligence • Data automatically saved")
