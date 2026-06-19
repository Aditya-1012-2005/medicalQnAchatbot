# 🩺 Medical QA Chatbot using QLoRA + Mistral-7B

An end-to-end medical question-answering system built using **Mistral-7B-Instruct**, **Unsloth**, and **QLoRA/LoRA fine-tuning**. The model is trained on curated medical datasets and optimized for efficient training on NVIDIA A100 GPUs.

---

## 🚀 Features

- 🤖 Fine-tuned **Mistral-7B-Instruct-v0.2**
- ⚡ Parameter-efficient training using **QLoRA / LoRA**
- 🧠 Uses **Unsloth** for accelerated training and inference
- 📚 Trained on multiple medical datasets
- 📊 Complete EDA and dataset analysis
- 📈 Training and validation loss visualization
- 📏 ROUGE and BERTScore evaluation
- 🩺 Interactive medical chatbot interface
- ☁️ Hugging Face Hub integration
- 🔄 Checkpoint resume support

---

## Model Architecture

```
Mistral-7B-Instruct-v0.2
        ↓
LoRA / RSLoRA Adapters
        ↓
Instruction Fine-Tuning
        ↓
Medical Question Answering
```

---

## Dataset

### 1. MedAlpaca MedicalQA
- Source: `medalpaca/medical_meadow_medqa`
- USMLE-style medical questions
- MCQ options removed during preprocessing

### 2. MedQuad NIH
- Source: `keivalya/MedQuad-MedicalQnADataset`
- Medical question-answer pairs from NIH

---

## Data Processing

### Cleaning Pipeline

- Remove HTML tags
- Remove duplicate samples
- Strip MCQ options
- Remove answer prefixes
- Length filtering
- Token filtering
- Merge datasets
- Shuffle and split train-validation sets

---

## Tech Stack

### Frameworks

- PyTorch
- Transformers
- Unsloth
- PEFT
- TRL

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Rouge Score
- BERTScore
- Datasets

---

## Training Configuration

| Parameter | Value |
|------------|-------|
| Base Model | Mistral-7B-Instruct-v0.2 |
| Max Sequence Length | 2048 |
| LoRA Rank | 64 |
| LoRA Alpha | 64 |
| RSLoRA | Enabled |
| Batch Size | 8 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 32 |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Precision | BF16 |
| Packing | Enabled |

---

## Project Structure

```bash
medical-qa-chatbot/
│
├── Medical_QA_QLoRA.ipynb
├── training_loss_v3.png
├── dataset_eda_v3.png
├── wordclouds.png
├── score_distributions.png
│
├── medical_qa_mistral_lora_v3/
│     ├── checkpoints
│     └── saved adapters
│
├── medical_qa_mistral_lora_v3_final/
│
├── medical_qa_mistral_merged_v3/
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/medical-qa-chatbot.git

cd medical-qa-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```python
trainer.train()
```

Resume from checkpoint:

```python
trainer.train(
    resume_from_checkpoint="./checkpoint-500"
)
```

---

## Inference

```python
answer = ask_medical_chatbot(
    "What are the symptoms of Type 2 Diabetes?"
)

print(answer)
```

---

## Evaluation Metrics

The model is evaluated using:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

Visualizations include:

- Loss curves
- Word clouds
- Dataset statistics
- Score distributions

---

## Example Questions

### Q:
> What are the early warning signs of Type 2 Diabetes?

### Q:
> What medications are commonly used to treat hypertension?

### Q:
> What are the symptoms of myocardial infarction?

### Q:
> What causes Alzheimer's disease?

---

## Saving Models

### LoRA adapters

```python
model.save_pretrained()
```

### Merged model

```python
model.save_pretrained_merged(
    save_method="merged_16bit"
)
```

### GGUF Export

```python
model.save_pretrained_gguf(
    quantization_method="q4_k_m"
)
```

---

## Results

✅ Efficient fine-tuning with Unsloth

✅ Reduced memory requirements using LoRA

✅ Stable training with RSLoRA

✅ High-quality medical responses

✅ Interactive chatbot support

---

## Future Improvements

- Retrieval-Augmented Generation (RAG)
- Multi-turn conversation support
- Clinical note summarization
- Vector database integration
- Streamlit deployment
- Quantized GGUF models for Ollama
- Multi-modal medical assistance

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.

The model should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## Author

**Aditya Santosh**

B.E. Artificial Intelligence & Data Science  
Minor in Cybersecurity  
VESIT Mumbai

---

## License

MIT License

---

⭐ If you found this project useful, consider giving it a star!
