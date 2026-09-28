# 🔬 FineTuneLab: Quantifiable Parameter-Efficient Fine-Tuning

An end-to-end machine learning pipeline and interactive Streamlit dashboard designed to quantify the trade-offs between Zero-Shot inference, Plain LoRA (FP16), and QLoRA (4-bit) fine-tuning.

Built around the `Qwen/Qwen2.5-0.5B` architecture, this project benchmarks Parameter-Efficient Fine-Tuning (PEFT) techniques specifically on the task of Python error diagnosis, analyzing structural adherence, code validity, and hardware resource consumption.

## 🚀 Key Features

* **Interactive Multi-Arm Inference:** A side-by-side testing playground that dynamically routes prompts to a Zero-Shot base model, a Plain LoRA adapter, and a QLoRA adapter, allowing for real-time qualitative comparison of hallucinations and structural generation.
* **Isolated State Management:** Implements dynamic PEFT adapter injection and strict `.unload()` garbage collection to prevent PyTorch shared model in-memory leaks during concurrent Streamlit generations.
* **Empirical Hardware Benchmarking:** Automated tracking of Peak VRAM (MB), training duration, adapter disk footprint, and exact-match syntax validation utilizing Python's `ast` parser.
* **LoRA Rank Ablation Analysis:** Comprehensive ablation study mapping the scaling laws of adapter rank ($r \in \{4, 8, 16, 32\}$) against trainable parameter count and validation loss convergence.

## 📊 Benchmark Highlights

* **Memory Efficiency:** QLoRA (4-bit) reduced Peak VRAM consumption by >60% compared to Plain LoRA (FP16) during the training phase, while maintaining identical structural accuracy and valid AST parsing rates.
* **Rank Ablation Insight:** Trainable parameter counts scale linearly with rank ($r$). However, because the adapter weight matrices ($\Delta W = B \cdot A$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$) account for under 2% of the frozen base parameter count, peak training VRAM remains dominated by base model activation memory and KV-caching. Task accuracy converges rapidly once $r \ge 8$.
* **Format Adherence:** Both LoRA and QLoRA successfully learned the strict target schema (`Cause:`, `Explanation:`, `Fix:`, `Corrected Code:`) via causal language modeling, whereas the Base Model reverted to unstructured conversational prose.

## 📁 Repository Structure

```text
FineTuneLab/
├── configs/
│   ├── base.yaml
│   ├── lora.yaml
│   └── qlora_r16.yaml
├── data/
│   └── train_test_split/
├── outputs/
│   ├── adapters/
│   │   ├── lora_r16/
│   │   └── qlora_r16/
│   └── metrics/
│       ├── ablation_ranks.json
│       └── benchmark_summary.json
├── src/
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── training/
│   │   └── run_ablation.py
│   └── utils/
├── app.py
└── requirements.txt

```

## 🛠️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/kousumi04/FineTuneLab.git
cd FineTuneLab

```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

**3. Launch the Dashboard**

```bash
streamlit run app.py

```

*Note: Inference executes dynamically. A CUDA-enabled GPU is recommended, but the dashboard automatically falls back to CPU execution (`torch.float32`) for adapter inference if no GPU is detected.*

## ⚙️ Tech Stack

* **Modeling & Fine-Tuning:** PyTorch, Hugging Face `transformers`, `peft`, `trl`, `bitsandbytes`
* **Data Processing:** `datasets`, `pandas`
* **Frontend:** Streamlit, Altair

---
