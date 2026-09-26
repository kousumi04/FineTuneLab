import streamlit as st
import torch
import ast
import json
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="FineTuneLab | Benchmark & Playground", page_icon="🔬", layout="wide")

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
BENCHMARK_PATH = "outputs/metrics/benchmark_summary.json"
ABLATION_PATH = "outputs/metrics/ablation_ranks.json"

ADAPTER_PATHS = {
    "Base Model (Zero-Shot)": None,
    "Plain LoRA (r=16)": "outputs/adapters/lora_r16",
    "QLoRA (r=16, 4-bit)": "outputs/adapters/qlora_r16"
}

@st.cache_resource(show_spinner="Loading models into memory...")
def get_base_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype, device_map=device)
    return tokenizer, model

def validate_syntax(code_string: str) -> bool:
    if not code_string.strip():
        return False
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False

def extract_corrected_code(text: str) -> str:
    marker = "Corrected Code:"
    return text.split(marker)[-1].strip() if marker in text else ""

st.title("🔬 FineTuneLab: Base vs. LoRA vs. QLoRA")
st.markdown("Quantifiable Parameter-Efficient Fine-Tuning Benchmarks on `Qwen/Qwen2.5-0.5B`.")

tab_inference, tab_metrics, tab_ablation = st.tabs([
    "🧪 Model Comparison Playground",
    "📊 Empirical Evaluation Table",
    "📈 LoRA Rank Ablation"
])

# ----------------- TAB 1: 3-WAY INFERENCE -----------------
with tab_inference:
    st.subheader("Interactive Multi-Arm Inference")
    col_ctrl1, col_ctrl2 = st.columns([1, 1])

    with col_ctrl1:
        preset = st.selectbox(
            "Select Preset Case:",
            ["Custom", "TypeError: String + Int", "IndexError: Array Bounds", "KeyError: Missing Dict Key"]
        )
        code_map = {
            "TypeError: String + Int": ("val = 'Age: ' + 30\nprint(val)", "TypeError: can only concatenate str (not 'int') to str"),
            "IndexError: Array Bounds": ("arr = [1, 2, 3]\nprint(arr[7])", "IndexError: list index out of range"),
            "KeyError: Missing Dict Key": ("data = {'id': 101}\nprint(data['token'])", "KeyError: 'token'"),
            "Custom": ("x = 10\ny = '20'\nprint(x + y)", "TypeError: unsupported operand type(s) for +: 'int' and 'str'")
        }
        raw_code, raw_err = code_map[preset]
        user_code = st.text_area("Faulty Code:", value=raw_code, height=110)
        user_err = st.text_input("Reported Error:", value=raw_err)

    with col_ctrl2:
        selected_arms = st.multiselect(
            "Choose Arms to Compare:",
            options=list(ADAPTER_PATHS.keys()),
            default=["Base Model (Zero-Shot)", "QLoRA (r=16, 4-bit)"]
        )
        run_btn = st.button("🚀 Run Diagnosis", type="primary", width="stretch")

    if run_btn and selected_arms:
        tokenizer, base_model = get_base_pipeline()
        prompt = f"### Instruction:\nDiagnose and fix the following Python error.\n\n### Code:\n{user_code}\n\n### Error:\n{user_err}\n\n### Response:\n"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        cols = st.columns(len(selected_arms))
        for idx, arm_name in enumerate(selected_arms):
            with cols[idx]:
                st.markdown(f"#### {arm_name}")
                adapter_path = ADAPTER_PATHS[arm_name]

                with st.spinner(f"Generating ({arm_name})..."):
                    if adapter_path is None:
                        # Raw Base Model
                        with torch.no_grad():
                            out = base_model.generate(**inputs, max_new_tokens=140, pad_token_id=tokenizer.eos_token_id, temperature=0.1)
                    else:
                        if os.path.exists(adapter_path):
                            peft_m = PeftModel.from_pretrained(base_model, adapter_path)
                            peft_m.eval()
                            with torch.no_grad():
                                out = peft_m.generate(**inputs, max_new_tokens=140, pad_token_id=tokenizer.eos_token_id, temperature=0.1)
                            base_model = peft_m.unload()
                        else:
                            st.error(f"Missing weights: `{adapter_path}`")
                            continue

                resp = tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()
                st.code(resp, language="markdown")

                code_fixed = extract_corrected_code(resp)
                if code_fixed:
                    if validate_syntax(code_fixed):
                        st.success("✅ AST Verified: Syntax is valid.")
                    else:
                        st.error("❌ AST Parse Failure: Syntax error found.")

# ----------------- TAB 2: BENCHMARK TABLE -----------------
with tab_metrics:
    st.subheader("System Benchmarks & Task Metrics (Saved Run Artifacts)")
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["Val Loss"] = df["Val Loss"].astype(str)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.warning(f"No benchmark file found at `{BENCHMARK_PATH}`. Run `python src/evaluation/evaluate_all.py` first.")

# ----------------- TAB 3: RANK ABLATION -----------------
with tab_ablation:
    st.subheader("LoRA Rank Ablation Analysis (r = 4, 8, 16, 32)")
    if os.path.exists(ABLATION_PATH):
        with open(ABLATION_PATH, "r") as f:
            abl_data = json.load(f)
        adf = pd.DataFrame(abl_data)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Rank vs. Trainable Parameters**")
            st.bar_chart(adf.set_index("rank")["trainable_params"])
        with c2:
            st.markdown("**Rank vs. Peak VRAM (MB)**")
            st.line_chart(adf.set_index("rank")["peak_vram_mb"])
        with c3:
            st.markdown("**Rank vs. Validation Loss**")
            st.line_chart(adf.set_index("rank")["val_loss"])

        st.dataframe(adf, width="stretch", hide_index=True)

        st.markdown(r"""
        > **Ablation Insight**: Trainable parameter count scales linearly with rank ($r$). However, because the adapter weight matrices ($\Delta W = B \cdot A$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$) account for under 2% of the frozen base parameter count, peak training VRAM remains largely dominated by base model activation memory and KV-caching. Task accuracy converges rapidly once $r \ge 8$.
        """)
    else:
        st.warning(f"No ablation data found at `{ABLATION_PATH}`. Run `python src/training/run_ablation.py`.")
