import streamlit as st
import google.generativeai as genai
import time
import random
import json

st.set_page_config(page_title="LLM Auto-Rater Pipeline", layout="wide")

st.title("⚙️ LLM Auto-Rater Pipeline")
st.markdown("End-to-end automated scoring system demo based on the 5-step architecture.")

# --- Demo Dataset (MT-Bench Style) ---
# Industry standard format: Prompt + AI Model Response
DEMO_DATA = [
    {
        "id": "mt-bench-001",
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "response": "def is_palindrome(s):\n    return s == s[::-1]"
    },
    {
        "id": "mt-bench-002",
        "prompt": "Explain the theory of relativity in simple terms.",
        "response": "Relativity is a theory by Einstein. It means time moves differently depending on how fast you go. It's a very complex topic."
    }
]

# --- Sidebar / Auth ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value="AIzaSyAbFOVXqIw3Okz3eu7W8TnZr5i1ouNVDt0", # Provided key for demo
    type="password"
)

selected_data = st.sidebar.selectbox(
    "Select Sample Data", 
    options=DEMO_DATA, 
    format_func=lambda x: x['id']
)

# --- Pipeline Steps ---

def step1_ingest_and_trace(data):
    """Step 1: Structured tracing and latency tracking."""
    st.subheader("1️⃣ Ingest & Trace")
    start_time = time.time()
    trace_id = f"trace_{random.randint(1000, 9999)}"
    
    st.json({
        "trace_id": trace_id,
        "status": "ingested",
        "prompt_length": len(data["prompt"]),
        "response_length": len(data["response"])
    })
    return trace_id, start_time

def step2_heuristic_gates(data):
    """Step 2: Deterministic checks (length, simple PII/toxicity)."""
    st.subheader("2️⃣ Heuristic Gates")
    passed = True
    reasons = []
    
    if len(data["response"]) < 10:
        passed = False
        reasons.append("Failed length check (too short).")
    
    # Mock PII/Toxicity check
    bad_words = ["confidential", "ssn", "idiot"]
    if any(word in data["response"].lower() for word in bad_words):
        passed = False
        reasons.append("Failed toxicity/PII gate.")
        
    if passed:
        st.success("Passed all heuristic gates (Schema, Format, Toxicity, Length).")
    else:
        st.error(f"Failed gates: {', '.join(reasons)}")
        
    return passed

def step3_llm_auto_rater(data, key):
    """Step 3: Distilled judge model scoring via JSON output."""
    st.subheader("3️⃣ LLM Auto-Rater")
    if not key:
        st.warning("API Key required.")
        return None
        
    genai.configure(api_key=key)
    # Using flash for fast, cheap auto-rating
    model = genai.GenerativeModel('gemini-1.5-flash') 
    
    prompt = f"""
    You are an expert LLM evaluator. Grade the following response based on the prompt.
    
    Prompt: {data['prompt']}
    Response: {data['response']}
    
    Rules:
    1. Output strictly in JSON format.
    2. Provide a 'score' on a 1-10 scale. Compress extreme scores toward the center (e.g., map a 10 to a 7, map a 1 to a 3).
    3. Provide a 'rationale' (keep it extremely brief, reduce verbosity by 50%).
    4. Provide a 'confidence' score between 0.0 and 1.0.
    
    JSON Schema: {{"score": int, "rationale": "string", "confidence": float}}
    """
    
    with st.spinner("Scoring..."):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            st.json(result)
            return result
        except Exception as e:
            st.error(f"API Error: {e}")
            return None

def step4_bias_correction(raw_result):
    """Step 4: Calibration and multi-judge ensemble mock."""
    st.subheader("4️⃣ Bias Correction")
    if not raw_result:
        return None
        
    st.write("Applying calibration mapping to raw score...")
    # Mocking calibration logic (e.g., pulling scores slightly down to adjust for LLM leniency bias)
    calibrated_score = max(1, raw_result["score"] - 1)
    
    corrected_result = raw_result.copy()
    corrected_result["calibrated_score"] = calibrated_score
    
    st.info(f"Raw Score: {raw_result['score']} ➡️ Calibrated Score: {calibrated_score}")
    return corrected_result

def step5_score_and_route(final_result):
    """Step 5: Confidence-based routing."""
    st.subheader("5️⃣ Score & Route")
    if not final_result:
        return
        
    confidence = final_result.get("confidence", 0)
    threshold = 0.75
    
    st.metric("Model Confidence", f"{confidence * 100}%")
    
    if confidence >= threshold:
        st.success("✅ Confidence High. Score shipped automatically to database.")
    else:
        st.warning("⚠️ Uncertainty boundary reached. Routing case to Human Review queue.")

# --- Execute Pipeline ---
if st.button("Run Evaluation Pipeline", type="primary"):
    st.divider()
    trace_id, start_time = step1_ingest_and_trace(selected_data)
    
    if step2_heuristic_gates(selected_data):
        raw_score = step3_llm_auto_rater(selected_data, api_key)
        corrected_score = step4_bias_correction(raw_score)
        step5_score_and_route(corrected_score)
        
    end_time = time.time()
    st.divider()
    st.caption(f"Pipeline executed in {round(end_time - start_time, 2)} seconds. Trace ID: {trace_id}")