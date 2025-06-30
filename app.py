import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import uuid
from datetime import datetime, date
import time

# Configure page
st.set_page_config(
    page_title="Hedge Fund Talent Map - SAFE MODE",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Hedge Fund Talent Map - CRASH-PROOF VERSION")

# Initialize minimal session state
if 'extractions' not in st.session_state:
    st.session_state.extractions = []

if 'people' not in st.session_state:
    st.session_state.people = []

# Simple Gemini setup
@st.cache_resource
def setup_gemini_safe(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Gemini setup failed: {e}")
        return None

# Safe file reading function
def read_file_safely(uploaded_file, max_size_kb=100):
    """Safely read uploaded file with size limits"""
    try:
        # Check file size
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        
        if file_size > max_size_kb * 1024:
            st.error(f"❌ File too large: {file_size/1024:.1f}KB. Max allowed: {max_size_kb}KB")
            return None
            
        st.info(f"📁 File size: {file_size/1024:.1f}KB")
        
        # Read file content
        raw_data = uploaded_file.getvalue()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = raw_data.decode(encoding)
                st.success(f"✅ File decoded with {encoding}")
                return content
            except UnicodeDecodeError:
                continue
                
        st.error("❌ Could not decode file with any encoding")
        return None
        
    except Exception as e:
        st.error(f"❌ File reading error: {e}")
        return None

# Simple extraction function
def extract_simple(text, model):
    """Simple extraction without complex processing"""
    try:
        # Limit text to prevent API issues
        if len(text) > 15000:
            text = text[:15000]
            st.warning(f"⚠️ Text truncated to 15,000 characters")
            
        prompt = f"""
Extract people and their career movements from this text. Return as JSON:

{text}

{{
  "people": [
    {{"name": "Full Name", "company": "Company", "role": "Position", "type": "hire/promotion/launch"}}
  ]
}}
"""
        
        with st.spinner("🤖 Processing with AI..."):
            response = model.generate_content(prompt)
            
        if not response or not response.text:
            st.error("❌ Empty response from AI")
            return []
            
        # Extract JSON
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1:
            st.error("❌ No JSON found in response")
            st.text_area("AI Response:", response_text, height=200)
            return []
            
        json_text = response_text[json_start:json_end]
        result = json.loads(json_text)
        
        return result.get('people', [])
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parsing error: {e}")
        return []
    except Exception as e:
        st.error(f"❌ Extraction error: {e}")
        return []

# SIDEBAR - Minimal AI Interface
with st.sidebar:
    st.header("🤖 AI Extraction")
    
    # API Key
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API key from secrets")
    except:
        api_key = st.text_input("Gemini API Key:", type="password")
    
    model = None
    if api_key:
        model = setup_gemini_safe(api_key)
        if model:
            st.success("✅ Model ready")
    
    st.markdown("---")
    
    # File Upload with Safety
    st.subheader("📁 Upload Newsletter")
    
    max_file_size = st.selectbox("Max file size:", [50, 100, 200], index=1, format_func=lambda x: f"{x}KB")
    
    uploaded_file = st.file_uploader(
        "Choose file:", 
        type=['txt'], 
        help=f"Max size: {max_file_size}KB"
    )
    
    newsletter_content = None
    
    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")
        
        # Safe file processing
        with st.expander("📊 File Info"):
            newsletter_content = read_file_safely(uploaded_file, max_file_size)
            
            if newsletter_content:
                char_count = len(newsletter_content)
                st.write(f"**Characters:** {char_count:,}")
                st.write(f"**Lines:** {newsletter_content.count(chr(10)) + 1}")
                
                # Show preview
                preview = newsletter_content[:500] + "..." if len(newsletter_content) > 500 else newsletter_content
                st.text_area("Preview:", preview, height=150)
    
    # Manual text input alternative
    st.markdown("---")
    st.subheader("✏️ Or Paste Text")
    manual_text = st.text_area("Newsletter text:", height=150, max_chars=10000)
    
    if manual_text:
        newsletter_content = manual_text
        st.info(f"📝 Manual text: {len(manual_text):,} characters")
    
    # Extract button
    if st.button("🚀 Extract Talent", use_container_width=True):
        if not newsletter_content:
            st.error("❌ No content to process")
        elif not model:
            st.error("❌ No API key or model")
        else:
            try:
                st.info("🔄 Starting extraction...")
                
                # Process with timeout
                start_time = time.time()
                extractions = extract_simple(newsletter_content, model)
                elapsed = time.time() - start_time
                
                if extractions:
                    # Add timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for ext in extractions:
                        ext['timestamp'] = timestamp
                    
                    # Add to session state
                    st.session_state.extractions.extend(extractions)
                    
                    st.success(f"✅ Found {len(extractions)} people in {elapsed:.1f}s")
                    st.rerun()
                else:
                    st.warning("⚠️ No extractions found")
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
    
    # Debug mode
    if st.checkbox("🐛 Debug mode"):
        st.write(f"**Session extractions:** {len(st.session_state.extractions)}")
        st.write(f"**Model loaded:** {model is not None}")
        st.write(f"**Content ready:** {newsletter_content is not None}")

# MAIN AREA - Simple Results Display
st.header("📊 Extraction Results")

if st.session_state.extractions:
    st.success(f"Found {len(st.session_state.extractions)} total extractions")
    
    # Display results
    for i, ext in enumerate(st.session_state.extractions):
        with st.expander(f"👤 {ext.get('name', 'Unknown')} → {ext.get('company', 'Unknown')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {ext.get('name', 'Unknown')}")
                st.write(f"**Company:** {ext.get('company', 'Unknown')}")
                st.write(f"**Role:** {ext.get('role', 'Unknown')}")
            
            with col2:
                st.write(f"**Type:** {ext.get('type', 'Unknown')}")
                st.write(f"**Extracted:** {ext.get('timestamp', 'Unknown')}")
            
            # Add to people database
            if st.button(f"➕ Add to Database", key=f"add_{i}"):
                new_person = {
                    "id": str(uuid.uuid4()),
                    "name": ext.get('name', 'Unknown'),
                    "current_title": ext.get('role', 'Unknown'),
                    "current_company_name": ext.get('company', 'Unknown'),
                    "location": "Unknown",
                    "email": "",
                    "phone": "",
                    "education": "",
                    "expertise": "",
                    "aum_managed": ""
                }
                st.session_state.people.append(new_person)
                st.success(f"✅ Added {ext.get('name')} to database")
                st.rerun()
    
    # Export functionality
    if st.button("📥 Export as CSV"):
        df = pd.DataFrame(st.session_state.extractions)
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "extractions.csv",
            "text/csv"
        )
    
    # Clear button
    if st.button("🗑️ Clear All Extractions"):
        st.session_state.extractions = []
        st.rerun()

else:
    st.info("👆 Upload a newsletter file or paste text in the sidebar to start extraction")
    
    # Test with sample
    if st.button("🧪 Test with Sample"):
        sample = """
        Harrison Balistreri launches Inevitable Capital Management.
        Sarah Gray joins Neil Chriss at Edge Peak.
        Daniel Crews promoted to deputy CIO at Tennessee Treasury.
        """
        
        if model:
            try:
                test_extractions = extract_simple(sample, model)
                if test_extractions:
                    st.session_state.extractions.extend(test_extractions)
                    st.success(f"✅ Test successful: {len(test_extractions)} people found")
                    st.rerun()
                else:
                    st.warning("⚠️ Test found no results")
            except Exception as e:
                st.error(f"❌ Test failed: {e}")
        else:
            st.error("❌ Setup API key first")

# Simple People Database View
if st.session_state.people:
    st.markdown("---")
    st.header("👥 People Database")
    
    people_data = []
    for person in st.session_state.people:
        people_data.append({
            "Name": person.get('name', 'Unknown'),
            "Title": person.get('current_title', 'Unknown'),
            "Company": person.get('current_company_name', 'Unknown')
        })
    
    if people_data:
        df = pd.DataFrame(people_data)
        st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.info("🔧 **SAFE MODE**: Simplified version to prevent crashes. Limited to basic extraction without complex batching.")
