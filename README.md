# 🚀 Faizan's LLM: Building a Large Language Model from scratch: 
**Then pretraining and fine-Tuning the Large Language Model for Classification and Instruction.**

## 🌟 Overview

This repository contains code and documentation for the Large Language Model I built from scratch. I then fine-tunied the LLM after pretraining the Transformer-based Large Language Models (LLMs). It covers essential topics such as:

- 🏆 Fine-tuning for classification and instruction-following tasks
- 📚 Pretraining a Transformer model from scratch
- 🎯 Low-Rank Adaptation (LoRA) for efficient fine-tuning
- 📉 Cosine decay learning rate scheduling
- 🚀 Gradient clipping for stable training
- ⚡ Detailed implementation of Transformer architectures

## 🔥 Features

- **🧠 Fine-Tuning Classification Models**: Train a Transformer-based model on a classification dataset.
- **📝 Fine-Tuning for Instruction Following**: Optimize models to follow instructions using reinforcement learning.
- **🛠️ Pretraining from Scratch**: Build and train a Transformer model with custom tokenization.
- **📊 LoRA Integration**: Implement LoRA for parameter-efficient fine-tuning.
- **📈 Cosine Decay Scheduler**: Adjust learning rate dynamically for smooth convergence.
- **🛡️ Gradient Clipping**: Prevent exploding gradients during training.
- **⚙️ Transformer Architecture**: Custom implementation of multi-head attention, layer normalization, and feed-forward networks.

## 📂 Folder Structure

```
├── finetuningclassification.ipynb  # Fine-tuning on classification tasks
├── finetuninginstruction.ipynb     # Fine-tuning for instruction following
├── pretraining.ipynb               # Pretraining a Transformer from scratch
├── transformer.ipynb               # Transformer architecture implementation
├── LLMcore.py                      # Core classes and functions for the LLM 
├── README.md                        # Documentation
```

## 🛠 Installation

Ensure you have the required dependencies installed before running the notebooks.

```bash
pip install torch transformers datasets
```
🔹 Fine-Tuning Classification ModelThe classification fine-tuning was performed using GPT-2 124M parameters, with pretrained weights loaded before fine-tuning.
Run the finetuningclassification.ipynb notebook to train a Transformer-based classifier.
🔹 Fine-Tuning for Instruction FollowingThe instruction fine-tuning was performed on GPT-2 300M medium parameters, using pretrained weights loaded before fine-tuning.
Use the finetuninginstruction.ipynb notebook to fine-tune an LLM for instruction-following tasks.
## 📌 Usage

### 🔹 Fine-Tuning Classification Model

Run the `finetuningclassification.ipynb` notebook to train a Transformer-based classifier.

### 🔹 Fine-Tuning for Instruction Following

Use the `finetuninginstruction.ipynb` notebook to fine-tune an LLM for instruction-following tasks.

### 🔹 Pretraining from Scratch

To pretrain a Transformer model from scratch, execute the `pretraining.ipynb` notebook.

### 🔹 Transformer Architecture

The `transformer.ipynb` notebook provides an in-depth implementation of Transformer blocks, including:

- 📌 Token and positional embeddings
- 📌 Multi-head self-attention
- 📌 Layer normalization
- 📌 Feed-forward networks
- 📌 Residual connections

## 📊 Training Details

### 🎯 Model Performance and Evaluation

- **✅ Accuracy Scores for the classification fine tuned LLM**: 
  - Training Accuracy: **94.81%**
  - Validation Accuracy: **96.53%**
  - Test Accuracy: **92.57%**
 
  -  **✅ Accuracy Scores for the instruction fine tuned LLM**: 
  - Accuracy Score: **45.84* as adjudicated by 'gpt 3.5 turbo' LLM model.
  - Room for improvement via modulation of the hyperparameters- learning rate, batch size, cosine decay and LoRA and model size.

  ![Training Accuracy](https://github.com/SYEDFAIZAN1987/Faizan-s-LLM--Building-an-LLM-from-scratch-and-fine-tuning-for-classification-and-instruction/blob/main/Accuracy%20score%20adjudicated%20by%20gpt%203.5%20turbo.png)

- **📉 Pretraining Loss Curve**:
  
  ![Pretraining Loss](https://github.com/SYEDFAIZAN1987/Faizan-s-LLM--Building-an-LLM-from-scratch-and-fine-tuning-for-classification-and-instruction/blob/main/Pretraining%20Loss.png)

- **🔥 Temperature Scaling in Pretraining**:
  
  ![Temperature Scaling](https://github.com/SYEDFAIZAN1987/Faizan-s-LLM--Building-an-LLM-from-scratch-and-fine-tuning-for-classification-and-instruction/blob/main/Temperature%20scaling%20in%20pretraining.png)

- **📊 Loss Curves for Classification Fine-Tuning**:
  
  ![Loss Curves](https://github.com/SYEDFAIZAN1987/Faizan-s-LLM--Building-an-LLM-from-scratch-and-fine-tuning-for-classification-and-instruction/blob/main/classification%20finetuning%20llm%20plot.png)

- **🔍 Classification Fine-Tuning Performance**:
  
  ![Classification Fine-Tuning](https://github.com/SYEDFAIZAN1987/Faizan-s-LLM--Building-an-LLM-from-scratch-and-fine-tuning-for-classification-and-instruction/blob/main/classification%20finetuning%20llm.png)

- **📝 Instruction Fine-Tuning Results**:
  
  ![Instruction Fine-Tuning](https://github.com/SYEDFAIZAN1987/Faizan-s-LLM--Building-an-LLM-from-scratch-and-fine-tuning-for-classification-and-instruction/blob/main/instruction%20finetuning%20llm%20plot.png)

### 📉 Cosine Decay Learning Rate

The learning rate is adjusted using cosine decay for stable convergence.

### 🚀 Gradient Clipping

To prevent instability, gradients are clipped during backpropagation.

### 🏆 Low-Rank Adaptation (LoRA)

LoRA is implemented to enable efficient fine-tuning with minimal computational cost.

## 📚 References

- 📖 Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
- 📖 Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.
- 📖 Hu, E. J., Wang, Y., Singh, A., Wang, Z., Yu, K., & Ainsworth, S. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- 📖 Sebastian Raschka. *Building an LLM from Scratch*.
- 📖 Jay Alammar and Maarten Grootendorst. Hands-On Large Language Models: Language Understanding and Generation.
- 📖 Andrej Karpathy. *Building a Chat-GPT*. Youtube
- 📖 Krish Naik. *Machine Learning and Deep Learning Tutorials*. Youtube and Udemy.

## 💡 Contributing

🎉 Contributions are welcome! Please feel free to submit issues or pull requests.

## 📜 License

This project is licensed under the **MIT License**.




