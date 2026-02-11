# End-to-End Automatic Speech Recognition (CTC-Based)

## Overview

This project implements a complete end-to-end Automatic Speech Recognition (ASR) pipeline using:

- Log-Mel Spectrogram feature extraction
- Character-level tokenization
- BiLSTM acoustic model
- Connectionist Temporal Classification (CTC) loss
- Greedy decoding
- CER / WER evaluation

The system is trained on Mozilla Common Voice (English - Australia subset).

This repository was developed as a research-focused implementation to understand alignment-free speech recognition from first principles.

---

## Architecture

Audio (.wav / .mp3 / .ogg)  
→ Log-Mel Spectrogram  
→ BiLSTM Encoder  
→ Linear Projection  
→ CTC Loss  
→ Greedy Decoder  
→ Text Output  

---

## Project Structure

speech-recognition-asr/  
│  
├── data/  
│ └── raw/commonvoice_en_au/  
│  
├── src/  
│ ├── dataset.py  
│ ├── features.py  
│ ├── tokenizer.py  
│ ├── model.py  
│ ├── collate.py  
│ ├── decode.py  
│ ├── metrics.py  
│ └── train.py  
│  
├── notebooks/  
│ ├── 01_dataset_exploration.ipynb  
│ ├── 02_logmel_extraction.ipynb  
│ ├── 03_text_processing.ipynb  
│ ├── 04_ctc_theory_and_loss.ipynb  
│ └── 05_asr_training.ipynb  
│  
├── scripts/  
│ └── run_pipeline.sh  
│  
├── graphs/  
└── requirements.txt  


---

## Training Details

- Features: 80-dimensional Log-Mel Spectrogram
- Encoder: 2-layer BiLSTM (hidden size 256)
- Loss: PyTorch CTC Loss
- Optimizer: Adam (lr=1e-3)
- Evaluation: Character Error Rate (CER), Word Error Rate (WER)

---

## Key Contributions

- Implemented CTC alignment from scratch
- Built tokenizer and decoding pipeline manually
- Structured ASR system without pretrained models
- Automated reproducible training pipeline

---

## Future Work

- Beam Search Decoding
- SpecAugment
- Conformer Encoder
- Subword Tokenization (BPE)
- Transformer-based ASR

---

## Author

Shivanshu Pal  
MSc Data Science  
Aspiring PhD Researcher (Speech / Audio AI)  
Email: contactshiva7@gmail.com