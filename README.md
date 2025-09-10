# AI Projects

Selected projects from my MSc. 
Code is private due to policy, however access to all private repositories and code can be made available upon request. Below are summaries and methods I used. 
All projects are implemented in Python & PyTorch.

## Deep Learning
MLP classifiers, CNN segmentation & depth and Transformer/CNN planners for autonomous driving.

### Model 1 — Image Classification (MLPs)
Training multi-layer perceptrons (MLPs) to classify images from the SuperTuxKart dataset into six object categories.  
- **Skills:** Training loops, TensorBoard, checkpointing  

### Model 2 — CNNs for Segmentation + Depth
Building convolutional networks to classify images and to perform road scene understanding (semantic segmentation + depth estimation).  
- **Skills:** Encoder–decoder CNNs, mIoU/MAE, augmentation  

### Model 3 — Planning (MLP / Transformer / CNN)
Predicting vehicle trajectories (future waypoints) from lane boundaries or directly from images for autonomous driving tasks.  
- **Skills:** Cross-attention, coordinate regression, offline driving metrics  


## Advanced Deep Learning
Advanced Deep Learning projects: memory-efficient networks, generative image models, language model reasoning for unit conversion, and vision-language models for Q&A on images.

### Model 1 — Low-Memory Training & Inference
Exploring techniques to reduce memory usage of large neural networks (BigNet, 73MB) without sacrificing performance.  
- **Skills:** Half-precision (FP16) training, LoRA/QLoRA adapters, 4-bit quantization, memory benchmarking, PyTorch custom layers  

### Model 2 — Auto-Regressive Image Generation
Developing a generative model for SuperTuxKart images, combining auto-encoders, binary spherical quantization, and autoregressive transformers.  
- **Skills:** Autoencoders, quantization (BSQ), sequence modeling with Transformers, generative sampling, compression  

### Model 3 — Unit Conversion with LLMs
Training and fine-tuning a language model (SmolLM2) to perform unit conversions, using in-context learning, supervised fine-tuning, and reinforcement learning.  
- **Skills:** Prompt engineering, chain-of-thought reasoning, LoRA fine-tuning, RLHF-style rejection fine-tuning  

### Model 4 — Vision-Language Model for SuperTuxKart
Building and fine-tuning a vision-language model (VLM) to answer questions about game images, with emphasis on constructing an effective data pipeline.  
- **Skills:** Vision-language pretraining, multimodal datasets, fine-tuning pipelines, evaluation of visual question answering  

## Natural Language Processing
Projects focused on sentiment analysis using the Rotten Tomatoes movie review dataset. Implemented in PyTorch and Python, starting from classical machine learning methods (such as Logistic Regression with unigram and bigram features) through to modern neural architectures.

Explored feature engineering (bag-of-words, improved n-gram representations), sparse vectorization, and optimization techniques. Later projects extended to deep learning with feedforward neural networks and GloVe word embeddings, implementing a Deep Averaging Network and experimenting with generalization to noisy data (e.g., misspellings).
- **Skills:** NLP, feature engineering, sentiment analysis  


