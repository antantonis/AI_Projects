# AI Projects

Selected projects from my MSc. Code is private due to policy, however access to all private repositories can be made available upon request. Below are summaries, methods, and results. 
All projects are implemented in Python & PyTorch.

## Deep Learning
MLP classifiers, CNN segmentation & depth and Transformer/CNN planners for autonomous driving.

### Assignment 1 — PyTorch & Classic ML Foundations
Practice with PyTorch tensors and implementing basic ML methods, including k-NN classification and simple time-series forecasting.  
- **Skills:** TorchScript-friendly ops, k-NN, simple regression  
- **Files:** `pytorch_basics.py`, `nearest_neighbor_classifier.py`, `weather_forecast.py`

### Assignment 2 — Image Classification (MLPs)
Training multi-layer perceptrons (MLPs) to classify images from the SuperTuxKart dataset into six object categories.  
- **Skills:** Training loops, TensorBoard, checkpointing  

### Assignment 3 — CNNs for Segmentation + Depth
Building convolutional networks to classify images and to perform road scene understanding (semantic segmentation + depth estimation).  
- **Skills:** Encoder–decoder CNNs, mIoU/MAE, augmentation  

### Assignment 4 — Planning (MLP / Transformer / CNN)
Learning to predict vehicle trajectories (future waypoints) from lane boundaries or directly from images for autonomous driving tasks.  
- **Skills:** Cross-attention, coordinate regression, offline driving metrics  


## Advanced Deep Learning
Advanced Deep Learning projects: memory-efficient networks, generative image models, language model reasoning for unit conversion, and vision-language models for Q&A on images.

### Assignment 1 — Low-Memory Training & Inference
Exploring techniques to reduce memory usage of large neural networks (BigNet, 73MB) without sacrificing performance.  
- **Skills:** Half-precision (FP16) training, LoRA/QLoRA adapters, 4-bit quantization, memory benchmarking, PyTorch custom layers  

### Assignment 2 — Auto-Regressive Image Generation
Developing a generative model for SuperTuxKart images, combining auto-encoders, binary spherical quantization, and autoregressive transformers.  
- **Skills:** Autoencoders, quantization (BSQ), sequence modeling with Transformers, generative sampling, compression  

### Assignment 3 — Unit Conversion with LLMs
Training and fine-tuning a language model (SmolLM2) to perform unit conversions, using in-context learning, supervised fine-tuning, and reinforcement learning.  
- **Skills:** Prompt engineering, chain-of-thought reasoning, LoRA fine-tuning, RLHF-style rejection fine-tuning  

### Assignment 4 — Vision-Language Model for SuperTuxKart
Building and fine-tuning a vision-language model (VLM) to answer questions about game images, with emphasis on constructing an effective data pipeline.  
- **Skills:** Vision-language pretraining, multimodal datasets, fine-tuning pipelines, evaluation of visual question answering  

## Natural Language Processing
Projects focused on sentiment analysis using the Rotten Tomatoes movie review dataset. Implemented in PyTorch and Python, starting from classical machine learning methods (Perceptron and Logistic Regression with unigram and bigram features) through to modern neural architectures.
Explored feature engineering (bag-of-words, improved n-gram representations), sparse vectorization, and optimization techniques. Later assignments extended to deep learning with feedforward neural networks and GloVe word embeddings, implementing a Deep Averaging Network and experimenting with generalization to noisy data (e.g., misspellings).
- **Skills:** NLP, feature engineering, sentiment analysis  


