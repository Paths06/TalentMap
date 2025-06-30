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
            "aum_managed": "2.5B USD"
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
            "aum_managed": "1.8B USD"
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
            "aum_managed": "800M USD"
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
                "location": person['location']
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
def setup_gemini(api_key):
    """Setup Gemini AI model"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error setting up Gemini: {e}")
        return None

def extract_talent(newsletter_text, model):
    """Extract Asian hedge fund talent movements using Gemini AI"""
    prompt = f"""
You are an expert at extracting talent movements from Asian financial newsletters, specifically focused on hedge funds, private equity, and asset management.

Extract people and their career movements from this text, focusing on:
- Hedge funds, private equity firms, asset management companies
- Asian markets (Hong Kong, Singapore, Tokyo, Seoul, Mumbai, etc.)
- Senior roles (PM, CIO, MD, Partner, Analyst, etc.)

Text to analyze:
{newsletter_text[:2000]}

Return ONLY a valid JSON object in this exact format:
{{
  "extractions": [
    {{
      "name": "First Last",
      "company": "Company Name",
      "previous_company": "Previous Company (if mentioned)",
      "movement_type": "hire",
      "title": "Position Title",
      "location": "City, Country",
      "context": "Brief description"
    }}
  ]
}}

Rules:
- Only extract real person names (First + Last name)
- Movement types: hire, promotion, launch, departure, partnership
- Focus on hedge funds and asset management
- Include Asian locations when mentioned
- No job titles or acronyms as names
"""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            result = json.loads(json_text)
            return result.get('extractions', [])
        
        return []
    except Exception as e:
        st.error(f"Extraction error: {e}")
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
    
    # API Key Setup
    st.subheader("🔑 Configuration")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API key loaded")
    except:
        api_key = st.text_input("Gemini API Key", type="password", help="Get from: https://makersuite.google.com/app/apikey")
    
    if api_key:
        model = setup_gemini(api_key)
        
        st.markdown("---")
        st.subheader("📰 Extract from Newsletter")
        
        input_method = st.radio("Input method:", ["📝 Text", "📁 File"])
        
        newsletter_text = ""
        if input_method == "📝 Text":
            newsletter_text = st.text_area("Newsletter content:", height=200, placeholder="Paste Asian hedge fund newsletter content...")
        else:
            uploaded_file = st.file_uploader("Upload file:", type=['txt'])
            if uploaded_file:
                try:
                    raw_data = uploaded_file.read()
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            newsletter_text = raw_data.decode(encoding)
                            st.success(f"File loaded: {len(newsletter_text):,} chars")
                            break
                        except UnicodeDecodeError:
                            continue
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("🚀 Extract Talent", use_container_width=True):
            if newsletter_text.strip() and model:
                with st.spinner("🤖 Analyzing..."):
                    extractions = extract_talent(newsletter_text, model)
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
            Li Wei Chen joins Hillhouse Capital as Portfolio Manager from Sequoia Capital China.
            Akira Tanaka promoted to CIO at Millennium Singapore office.
            Sarah Kim moves from Goldman Sachs Asia to head research at Citadel Hong Kong.
            Zhang Ming launches new fund at Greenwoods Asset Management.
            Priya Sharma departs Two Sigma Asia for family office role.
            """
            if model:
                with st.spinner("Testing..."):
                    extractions = extract_talent(sample_text, model)
                    if extractions:
                        st.success(f"✅ Found {len(extractions)} movements!")
                        with st.expander("Results"):
                            for ext in extractions:
                                st.write(f"• **{ext['name']}** → {ext['company']}")
    
    # Recent Extractions
    if st.session_state.all_extractions:
        st.markdown("---")
        st.subheader("📊 Recent Extractions")
        st.metric("Total Extracted", len(st.session_state.all_extractions))
        
        for ext in st.session_state.all_extractions[-3:]:
            with st.expander(f"{ext['name']} → {ext['company']}"):
                st.write(f"**Type:** {ext['movement_type']}")
                if ext.get('title'):
                    st.write(f"**Title:** {ext['title']}")
                if ext.get('location'):
                    st.write(f"**Location:** {ext['location']}")
                if st.button(f"➕ Add {ext['name']}", key=f"add_{ext['name']}_{ext['timestamp']}"):
                    # Add to people and firms
                    new_person_id = str(uuid.uuid4())
                    st.session_state.people.append({
                        "id": new_person_id,
                        "name": ext['name'],
                        "current_title": ext.get('title', 'Unknown'),
                        "current_company_name": ext['company'],
                        "location": ext.get('location', 'Asia'),
                        "email": "",
                        "linkedin_profile_url": "",
                        "phone": "",
                        "education": "",
                        "expertise": "",
                        "aum_managed": ""
                    })
                    
                    # Add firm if doesn't exist
                    if not get_firm_by_name(ext['company']):
                        new_firm_id = str(uuid.uuid4())
                        st.session_state.firms.append({
                            "id": new_firm_id,
                            "name": ext['company'],
                            "location": ext.get('location', 'Asia'),
                            "headquarters": "Unknown",
                            "aum": "Unknown",
                            "founded": None,
                            "strategy": "Hedge Fund",
                            "website": "",
                            "description": f"Hedge fund operating in {ext.get('location', 'Asia')}"
                        })
                    
                    # Add employment
                    st.session_state.employments.append({
                        "id": str(uuid.uuid4()),
                        "person_id": new_person_id,
                        "company_name": ext['company'],
                        "title": ext.get('title', 'Unknown'),
                        "start_date": date.today(),
                        "end_date": None,
                        "location": ext.get('location', 'Asia')
                    })
                    
                    st.success(f"✅ Added {ext['name']}!")
                    st.rerun()

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
                    "aum_managed": aum
                })
                
                # Add employment record
                st.session_state.employments.append({
                    "id": str(uuid.uuid4()),
                    "person_id": new_person_id,
                    "company_name": company,
                    "title": title,
                    "start_date": start_date,
                    "end_date": None,
                    "location": location
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
                "Location": person['location'],
                "AUM Managed": person.get('aum_managed', ''),
                "Expertise": person.get('expertise', ''),
                "ID": person['id']
            })
        
        df = pd.DataFrame(people_data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            location_filter = st.selectbox("Filter by Location", ["All"] + list(df['Location'].unique()))
        with col2:
            company_filter = st.selectbox("Filter by Company", ["All"] + list(df['Company'].unique()))
        with col3:
            expertise_filter = st.text_input("Search by Expertise", placeholder="Technology, Healthcare...")
        
        # Apply filters
        filtered_df = df.copy()
        if location_filter != "All":
            filtered_df = filtered_df[filtered_df['Location'] == location_filter]
        if company_filter != "All":
            filtered_df = filtered_df[filtered_df['Company'] == company_filter]
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
                    st.write(f"**📧 Email:** {person['email']}")
                    st.write(f"**📱 Phone:** {person['phone']}")
                    st.write(f"**🎓 Education:** {person['education']}")
                with col2:
                    st.write(f"**🏆 Expertise:** {person['expertise']}")
                    st.write(f"**💰 AUM Managed:** {person['aum_managed']}")
                    if person['linkedin_profile_url']:
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
        if person['email']:
            st.markdown(f"📧 [{person['email']}](mailto:{person['email']})")
        if person['phone']:
            st.markdown(f"📱 {person['phone']}")
        if person['linkedin_profile_url']:
            st.markdown(f"🔗 [LinkedIn Profile]({person['linkedin_profile_url']})")
    
    with col2:
        if person['education']:
            st.markdown(f"🎓 **Education:** {person['education']}")
        if person['expertise']:
            st.markdown(f"🏆 **Expertise:** {person['expertise']}")
        if person['aum_managed']:
            st.markdown(f"💰 **AUM Managed:** {person['aum_managed']}")
    
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
