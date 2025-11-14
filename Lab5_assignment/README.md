# 📘 Knowledge Distillation of Phi-3 Mini Using Unsloth  
### Bridging Classical CNN-Based KD → Modern LLM Distillation

This project implements **knowledge distillation (KD)** on a **Large Language Model (LLM)** using **Unsloth**, inspired by the conceptual workflow in a classical **CNN teacher–student KD notebook**.  

In the professor’s notebook, a CNN teacher model distills knowledge into a smaller CNN student via logits and KL-divergence.  
Here, the same KD idea is adapted to LLMs by distilling **chain-of-thought (CoT) reasoning** instead of logits.

---

## 🧠 1. Background: Classical KD vs LLM KD

| Classical CNN KD | LLM KD (This Project) |
|------------------|------------------------|
| Teacher outputs **logits** | Teacher outputs **CoT text** |
| Student learns via **KL + CE** | Student learns via **CE on text tokens** |
| Outputs fixed class vectors | Outputs free-form natural language |
| Requires labeled dataset | Uses **teacher-generated dataset** |
| Small models (CNNs) | LLMs (Phi-3 Mini) |

**Key insight:**  
> For LLMs, distilling the *reasoning text* is more practical and powerful than distilling raw logits.

---

## 🏗️ 2. Project Overview

This project performs end-to-end KD:

1. **Seed Question Collection**  
   ~50 reasoning questions covering math, logic, comprehension, etc.

2. **Teacher Model Data Generation**  
   Using GPT-4o-mini to generate:  
   - Original solutions (CoT)  
   - New transformed questions  
   - New CoT solutions  

   Expanded to **1,000 examples**.

3. **Dataset Formatting**  
   Converted into instruction-style format:  

Q: 
A: 

4. **Train/Validation Split**  
- 900 samples → train  
- 100 samples → validation  

5. **Fine-Tuning Phi-3 Mini with Unsloth**  
Using:  
- 4-bit quantized model  
- LoRA (rank 8)  
- Gradient checkpointing  
- Sequence length 1024  
- SFTTrainer (supervised fine-tuning)  

6. **Evaluation**  
- Validation loss  
- Perplexity  
- Token accuracy (optional)  
- Teacher vs Student reasoning comparison  

7. **Analysis**  
Student model demonstrated strong reasoning alignment with teacher.

---

## 📚 3. Dataset Construction

### Step 1 — 50 Seed Questions  
Provided manually.

### Step 2 — Teacher Model Expansion  
For each seed:
- Generate chain-of-thought solution  
- Generate similar new question  
- Generate new CoT solution  

### Step 3 — Final Dataset Structure

```json
{
"original_question": "...",
"original_solution": "...",
"new_question": "...",
"new_solution": "..."
}
```

Step 4 — Training Format

Q: <question>
A: <chain-of-thought from teacher>


⸻

⚙️ 4. Fine-Tuning Setup (Unsloth + Phi-3 Mini)

Model Used

unsloth/Phi-3-mini-4k-instruct-bnb-4bit

Config
	•	Max sequence length: 1024
	•	LoRA rank: 8
	•	Batch size: 1 (grad accumulation = 4)
	•	Epochs: 2–4
	•	Gradient checkpointing: enabled
	•	Optimizer & training: handled by SFTTrainer

Training Snippet

```
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


⸻

📊 5. Evaluation Results

Quantitative

Epoch	Train Loss	Val Los 
    1	 0.568	     0.618	       
    2	 0.424	     0.527	       

Interpretation:
	•	Loss decreasing → healthy learning
	•	Perplexity < 2 → very strong for small KD dataset
	•	Token accuracy is low → expected (free-form text generation task)

Qualitative

Teacher vs Student generations show:
	•	Correct reasoning
	•	Strong structural similarity
	•	Student reaches correct answers
	•	Coherent step-by-step logic
	•	Minor repetition (fixed through decoding parameters)

The student model successfully mimics the teacher on unseen test questions.

⸻

🔍 6. How This Bridges the Professor’s KD Notebook

Mapping professor’s CNN KD → LLM KD:

Professor CNN Step	LLM KD Equivalent
Teacher computes logits	Teacher LLM generates CoT reasoning
Student learns from KL + CE	Student learns via CE only (text)
Use DataLoader	Use HuggingFace Datasets + SFTTrainer
Evaluate accuracy	Evaluate loss + perplexity + CoT comparison
Logits → class predictions	Text → reasoning + answer

The structure of KD is identical:

Teacher → Supervision → Student → Evaluation

Only the modality (images vs text) differs.

⸻

🎯 7. Key Takeaways
	•	LLM KD works by distilling reasoning traces, not logits.
	•	Unsloth enables low-VRAM fine-tuning in Google Colab.
	•	Phi-3 Mini is an effective student model for KD.
	•	Student learned clear reasoning structure and accuracy.
	•	Evaluation must be both quantitative and qualitative.

⸻

📁 8. Folder Structure

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

