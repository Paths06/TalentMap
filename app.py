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

st.title("🏢 Hedge Fund Talent Map - STABLE VERSION")

# Quick mode comparison
with st.container():
    st.success("""
    ✅ **CRASH-PROOF EXTRACTION:**
    • **🛡️ Safe Mode**: Process first 15K chars instantly (good for testing)
    • **⚡ Chunked Mode**: Process full file safely in 12K chunks (complete extraction)
    • **🔄 Progress Tracking**: See exactly what's happening
    • **🚫 No Crashes**: File size limits and error recovery
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🛡️ Safe Mode:** Fast • 15K chars • ~8 people")
    with col2:
        st.info("**⚡ Chunked Mode:** Complete • 50K chars • ~20+ people")
    with col3:
        st.info("**🔍 Debug Mode:** Shows chunk details & AI responses")

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

# Safe chunked extraction function with debugging
def extract_chunked_safe(text, model, chunk_size=12000, max_chunks=5):
    """Process large text in safe chunks with detailed debugging"""
    try:
        total_chars = len(text)
        st.info(f"📄 Processing {total_chars:,} characters in chunks of {chunk_size:,}")
        
        all_extractions = []
        processed_chars = 0
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        debug_container = st.expander("🐛 Debug Info (Click to see chunk details)")
        
        # Process in chunks
        chunk_num = 0
        while processed_chars < total_chars and chunk_num < max_chunks:
            chunk_num += 1
            
            # Extract chunk with small overlap
            start_pos = max(0, processed_chars - 500)  # 500 char overlap
            end_pos = min(start_pos + chunk_size, total_chars)
            chunk_text = text[start_pos:end_pos]
            
            with debug_container:
                st.write(f"**Chunk {chunk_num}:**")
                st.write(f"- Start: {start_pos:,}, End: {end_pos:,}")
                st.write(f"- Length: {len(chunk_text):,} chars")
                st.write(f"- Preview: {chunk_text[:200]}...")
                
            status_text.info(f"🔄 Processing chunk {chunk_num} ({len(chunk_text):,} chars)...")
            
            # Rate limiting delay
            if chunk_num > 1:
                status_text.info(f"⏱️ Rate limit delay: 3 seconds...")
                time.sleep(3)
            
            # Process chunk with detailed error handling
            try:
                with debug_container:
                    st.write(f"- Sending to AI...")
                
                chunk_extractions = extract_simple_debug(chunk_text, model, chunk_num)
                
                with debug_container:
                    st.write(f"- Raw AI result: {len(chunk_extractions)} extractions")
                    if chunk_extractions:
                        for ext in chunk_extractions:
                            st.write(f"  • {ext.get('name', 'NO_NAME')} → {ext.get('company', 'NO_COMPANY')}")
                
                if chunk_extractions:
                    # Simple deduplication by name (case insensitive)
                    existing_names = {ext.get('name', '').lower().strip() for ext in all_extractions if ext.get('name')}
                    
                    new_extractions = []
                    for ext in chunk_extractions:
                        name = ext.get('name', '').lower().strip()
                        if name and name not in existing_names:
                            new_extractions.append(ext)
                            existing_names.add(name)
                    
                    all_extractions.extend(new_extractions)
                    
                    with debug_container:
                        st.write(f"- After deduplication: {len(new_extractions)} new extractions")
                        st.write(f"- Total so far: {len(all_extractions)}")
                    
                    status_text.success(f"✅ Chunk {chunk_num}: Found {len(chunk_extractions)} people ({len(new_extractions)} new, {len(all_extractions)} total)")
                else:
                    with debug_container:
                        st.write(f"- ❌ No extractions found in this chunk")
                    status_text.warning(f"⚠️ Chunk {chunk_num}: No extractions found")
                
            except Exception as e:
                with debug_container:
                    st.write(f"- ❌ ERROR: {e}")
                status_text.error(f"❌ Chunk {chunk_num} failed: {e}")
                # Continue with next chunk
            
            # Update progress
            processed_chars = end_pos
            progress = min(processed_chars / total_chars, 1.0)
            progress_bar.progress(progress)
            
            # Safety break
            if chunk_num >= max_chunks:
                status_text.warning(f"⚠️ Stopped at {max_chunks} chunks (safety limit)")
                break
        
        # Final status
        progress_bar.progress(1.0)
        status_text.success(f"🎯 **Complete!** Processed {processed_chars:,}/{total_chars:,} chars in {chunk_num} chunks")
        
        with debug_container:
            st.write(f"**FINAL RESULT: {len(all_extractions)} total extractions**")
            
        return all_extractions
        
    except Exception as e:
        st.error(f"❌ Chunked processing failed: {e}")
        return []

# Debug version of extract_simple
def extract_simple_debug(text, model, chunk_num=0):
    """Simple extraction with debugging output"""
    try:
        # Limit text to prevent API issues
        original_length = len(text)
        if len(text) > 15000:
            text = text[:15000]
            
        prompt = f"""
Extract people and their career movements from this newsletter text. Return as JSON:

{text}

{{
  "people": [
    {{"name": "Full Name", "company": "Company", "role": "Position", "type": "hire/promotion/launch"}}
  ]
}}

Find EVERY person mentioned in professional contexts. Look for names like:
- Harrison Balistreri, Vince Ortiz, Robin Boldt
- Daniel Crews, Sarah Gray, Neil Chriss  
- Louis Couronne, Macaire Chue, Grant Leslie
- Any "X joins Y", "X launches Z", "X promoted to"
"""
        
        response = model.generate_content(prompt)
            
        if not response or not response.text:
            return []
            
        # Extract JSON
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1:
            # Show AI response for debugging
            st.error(f"Chunk {chunk_num}: No JSON found in AI response")
            st.text_area(f"AI Response Chunk {chunk_num}:", response_text, height=150)
            return []
            
        json_text = response_text[json_start:json_end]
        
        try:
            result = json.loads(json_text)
            return result.get('people', [])
        except json.JSONDecodeError as e:
            st.error(f"Chunk {chunk_num}: JSON parsing error: {e}")
            st.text_area(f"Invalid JSON Chunk {chunk_num}:", json_text, height=150)
            return []
        
    except Exception as e:
        st.error(f"Chunk {chunk_num}: Extraction error: {e}")
        return []
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
    
    processing_mode = st.radio(
        "Processing mode:",
        ["🛡️ Safe (15K chars)", "⚡ Chunked (Full file)", "🔍 Debug Chunked"],
        help="Safe: Fast single chunk. Chunked: Full file processing. Debug: Shows detailed chunk info."
    )
    
    if processing_mode.startswith("🛡️"):
        max_file_size = st.selectbox("Max file size:", [50, 100, 200], index=1, format_func=lambda x: f"{x}KB")
    else:
        max_file_size = st.selectbox("Max file size:", [100, 200, 500], index=1, format_func=lambda x: f"{x}KB")
        st.info("🔄 **Chunked mode**: Will process full file in safe 12K chunks with 3s delays")
    
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
                start_time = time.time()
                
                # Choose processing method
                if processing_mode.startswith("🛡️"):
                    # Safe mode - single chunk
                    st.info(f"🛡️ **Safe mode**: Processing first 15K characters")
                    extractions = extract_simple(newsletter_content, model)
                elif processing_mode.startswith("🔍"):
                    # Debug chunked mode
                    st.info(f"🔍 **Debug mode**: Processing full {len(newsletter_content):,} characters with detailed logging")
                    extractions = extract_chunked_safe(newsletter_content, model)
                else:
                    # Regular chunked mode
                    st.info(f"⚡ **Chunked mode**: Processing full {len(newsletter_content):,} characters")
                    extractions = extract_chunked_safe(newsletter_content, model)
                
                elapsed = time.time() - start_time
                
                if extractions:
                    # Add timestamp and mode
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    mode_icon = processing_mode.split()[0]
                    
                    for ext in extractions:
                        ext['timestamp'] = timestamp
                        ext['mode'] = mode_icon
                    
                    # Add to session state
                    st.session_state.extractions.extend(extractions)
                    
                    st.success(f"✅ Found {len(extractions)} people in {elapsed:.1f}s")
                    
                    # Show comparison
                    if processing_mode.startswith("🛡️"):
                        estimated_full = int(len(extractions) * (len(newsletter_content) / 15000))
                        st.info(f"💡 **Estimate**: Full file might contain ~{estimated_full} people. Try chunked mode for complete extraction.")
                    elif processing_mode.startswith("⚡") or processing_mode.startswith("🔍"):
                        st.info(f"🎯 **Full file processed**: {len(newsletter_content):,} characters analyzed")
                    
                    st.rerun()
                else:
                    st.warning("⚠️ No extractions found")
                    
                    # Provide debugging help
                    if processing_mode.startswith("⚡") or processing_mode.startswith("🔍"):
                        st.error("**Chunked mode found nothing!** This suggests:")
                        st.write("1. 🔍 Try 'Debug Chunked' mode to see what's happening")
                        st.write("2. 📄 Check if chunks contain the right content")  
                        st.write("3. 🤖 Verify AI responses in debug mode")
                        st.write("4. 🛡️ Safe mode worked, so extraction logic is fine")
                        
                        # Quick test suggestion
                        if st.button("🧪 Test First Chunk Only"):
                            test_chunk = newsletter_content[:12000]
                            st.write(f"Testing first 12K characters (same as chunk 1):")
                            test_extractions = extract_simple_debug(test_chunk, model, "TEST")
                            if test_extractions:
                                st.success(f"✅ First chunk test found {len(test_extractions)} people!")
                                st.write("This means chunking logic has a bug.")
                            else:
                                st.error("❌ Even first chunk test failed - check AI responses above")
                    else:
                        st.info("💡 **Tip**: Try safe mode if chunked mode fails")
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                st.info("💡 **Tip**: Try safe mode if chunked mode fails")
    
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
        mode_badge = ext.get('mode', '🔧')
        with st.expander(f"{mode_badge} {ext.get('name', 'Unknown')} → {ext.get('company', 'Unknown')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {ext.get('name', 'Unknown')}")
                st.write(f"**Company:** {ext.get('company', 'Unknown')}")
                st.write(f"**Role:** {ext.get('role', 'Unknown')}")
            
            with col2:
                st.write(f"**Type:** {ext.get('type', 'Unknown')}")
                st.write(f"**Mode:** {ext.get('mode', 'Unknown')}")
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
    
    # Results analysis
    with st.expander("📊 Results Analysis"):
        safe_mode_results = [ext for ext in st.session_state.extractions if ext.get('mode') == '🛡️']
        chunked_mode_results = [ext for ext in st.session_state.extractions if ext.get('mode') == '⚡']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🛡️ Safe Mode", len(safe_mode_results))
        with col2:
            st.metric("⚡ Chunked Mode", len(chunked_mode_results))
        
        if safe_mode_results and chunked_mode_results:
            st.success("✅ **Chunked mode found more people!** This shows full file processing works.")
        elif safe_mode_results:
            st.info("💡 Try chunked mode to process your full file and find more people.")
    
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
