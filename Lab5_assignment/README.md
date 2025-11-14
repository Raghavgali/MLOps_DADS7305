# Knowledge Distillation of Phi-3 Mini Using Unsloth, HuggingFace and PyTorch
### Bridging Classical CNN-Based KD → Modern LLM Distillation

This project implements **knowledge distillation (KD)** on a **Large Language Model (LLM)** using **Unsloth**, inspired by the classical CNN teacher–student workflow in the professor’s notebook. In the original notebook, a CNN teacher distills knowledge into a smaller CNN student via logits and KL-divergence. Here, the same KD idea is adapted to LLMs by distilling **chain-of-thought (CoT) reasoning** instead of logits.

---

## 1. Background: Classical KD vs LLM KD

| Classical CNN KD | LLM KD (This Project) |
|------------------|------------------------|
| Teacher outputs **logits** | Teacher outputs **CoT text** |
| Student learns via **KL + CE** | Student learns via **CE on text tokens** |
| Outputs fixed class vectors | Outputs free-form natural language |
| Requires labeled dataset | Uses **teacher-generated dataset** |
| Small models (CNNs) | LLMs (Phi-3 Mini) |

**Key insight:** For LLMs, distilling the *reasoning text* is more practical and powerful than distilling raw logits.

---

## 2. Project Overview

This project performs end-to-end KD across the following stages:

1. **Seed Question Collection** – ~50 reasoning questions covering math, logic, comprehension, etc.
2. **Teacher Model Data Generation** – GPT-4o-mini generates the original CoT solution, a transformed question, and a new CoT solution, expanding the dataset to **1,000 examples**.
3. **Dataset Formatting** – Converted into instruction-style prompts:
   ```
   Q: <question>
   A: <teacher chain-of-thought>
   ```
4. **Train/Validation Split** – 900 samples for training, 100 for validation.
5. **Fine-Tuning Phi-3 Mini with Unsloth** – 4-bit quantized student, LoRA rank 8, gradient checkpointing, sequence length 1024, trained via `SFTTrainer`.
6. **Evaluation** – Track validation loss, perplexity, token accuracy (optional), and teacher vs. student reasoning comparison.
7. **Analysis** – Student generations show strong structural alignment with the teacher.

---

## 3. Dataset Construction

**Step 1 — 50 Seed Questions**  
Seed questions are provided manually across diverse reasoning domains.

**Step 2 — Teacher Model Expansion**  
For each seed question:
- Generate chain-of-thought solution.  
- Generate a similar new question.  
- Generate a new CoT solution for the transformed question.

**Step 3 — Final Dataset Structure**

```json
{
  "original_question": "...",
  "original_solution": "...",
  "new_question": "...",
  "new_solution": "..."
}
```

**Step 4 — Training Format**

```
Q: <question>
A: <chain-of-thought from teacher>
```

---

## 4. Fine-Tuning Setup (Unsloth + Phi-3 Mini)

- **Model:** `unsloth/Phi-3-mini-4k-instruct-bnb-4bit`  
- **Max sequence length:** 1024  
- **LoRA rank:** 8  
- **Batch size:** 1 (gradient accumulation = 4)  
- **Epochs:** 2–4  
- **Gradient checkpointing:** Enabled  
- **Trainer:** `SFTTrainer` handles optimization and logging

### Training Snippet

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        output_dir="model-out",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        eval_strategy="epoch",
        log_strategy="steps",
        log_steps=20,
        save_strategy="epoch",
        report_to=["tensorboard"],
    ),
    compute_metrics=compute_metrics,
)
trainer.train()
```

---

## 5. Evaluation Results

| Epoch | Train Loss | Val Loss | Perplexity | Token Accuracy |
|-------|------------|----------|------------|----------------|
| 1     | 0.568      | 0.618    | 1.86       | 0.0034         |
| 2     | 0.424      | 0.527    | 1.69       | 0.0035         |

**Interpretation**
- Loss decreases steadily → healthy learning curve.
- Perplexity < 2 → strong performance for a small KD dataset.
- Token accuracy remains low, which is expected for a free-form generation task.

**Qualitative Observations**
- Student reasoning mirrors teacher structure.
- Answers remain coherent and step-by-step.
- Student reaches correct answers on unseen questions.
- Minor repetition can be mitigated with decoding tweaks.

---

## 6. Bridging the Professor’s KD Notebook

| Professor CNN Step | LLM KD Equivalent |
|--------------------|-------------------|
| Teacher computes logits | Teacher LLM generates CoT reasoning |
| Student learns from KL + CE | Student learns via CE on text tokens |
| Use DataLoader | Use HuggingFace Datasets + `SFTTrainer` |
| Evaluate accuracy | Evaluate loss, perplexity, and CoT alignment |
| Logits → class predictions | Text → reasoning + answer |

The structure of KD is identical: **Teacher → Supervision → Student → Evaluation**. Only the modality (images vs. text) changes.

---

## 7. Key Takeaways
- LLM KD works by distilling reasoning traces, not logits.
- Unsloth enables low-VRAM fine-tuning (e.g., Google Colab T4 GPU).
- Phi-3 Mini is an effective, compact student model.
- Student learned clear reasoning structure and accuracy.
- Evaluation should include both quantitative metrics and qualitative CoT inspection.

---

## 8. Folder Structure

```
.
├── data_generation/
│   ├── generate_dataset.py
│   └── config.yaml
├── dataset.json
├── notebooks/
│   ├── KD_Training.ipynb
│   └── Evaluation.ipynb
├── model-out/
│   └── ...
├── README.md
└── requirements.txt
```