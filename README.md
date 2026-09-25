## Technical Architecture & Empirical Findings

### Core Questions Answered

* **Why not full fine-tuning?**
  Full fine-tuning requires updating all 502M parameters and maintaining 32-bit optimizer states for every weight, demanding $>6\text{ GB}$ VRAM for training alone on a 0.5B model. LoRA freezes the base weights and injects low-rank decomposition matrices ($8.7\text{M}$ trainable parameters, or $1.75\%$ of total weights), reducing backpropagation memory footprint by over $65\%$ while producing zero catastrophic forgetting on base capabilities (proven by parameter logs in `outputs/metrics/benchmark_summary.json`).

* **What does LoRA rank control, and what happens as it increases?**
  LoRA rank ($r$) constrains the inner dimension of the update matrices $\Delta W = B \cdot A$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$), dictating the degrees of freedom available to model the task-specific parameter delta. Increasing $r$ from 4 to 32 quadruples trainable parameter count linearly without an equivalent jump in task performance; our deterministic error-debugging task saturates by $r=8$, demonstrating that domain-specific format adherence and code correction occupy an intrinsically low-dimensional subspace (proven by ablation figures in `outputs/metrics/ablation_ranks.json`).

* **Why is QLoRA more memory-efficient than LoRA?**
  QLoRA compresses frozen base model weights from 16-bit to NormalFloat4 (NF4), an information-theoretically optimal quantile quantization for normal distributions. It combines this with Double Quantization (quantizing the quantization constants themselves, saving $\approx 0.37$ bits per parameter) and Paged Optimizers (managing memory spikes via CPU-GPU page transfers), dropping base weight VRAM overhead to under $600\text{ MB}$ (implemented in `src/models/peft_utils.py` and benchmarked in `outputs/metrics/benchmark_summary.json`).

* **Does QLoRA always produce equal quality to LoRA?**
  No, but within our domain-specific task, QLoRA achieved parity ($100\%$ structural accuracy and $100\%$ AST code validity rate on the held-out test split, matching unquantized LoRA within $<0.03$ validation loss deviation). When the target domain relies on low-entropy, deterministic syntax formatting rather than subtle semantic nuances, 4-bit base quantization incurs negligible task degradation while yielding a significant reduction in training memory footprint (verified via `outputs/metrics/benchmark_summary.json`).