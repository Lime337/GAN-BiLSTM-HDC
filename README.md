# GAN–BiLSTM–HDC: A Hybrid Framework for Robust and Hardware-Efficient Malware Detection
# Date of Creation: 8/11/2025

---

## Methodology
1. **Generative Modeling:**  
   Two GAN architectures are trained to generate benign and malware instruction sequences. The baseline GAN (`GAN_model.py`) ensures syntactic realism, while a second GAN variant (`GAN_model_backprop.py`) incorporates feedback from a semantic critic to improve semantic consistency.

2. **Semantic Critic:**  
   A BiGRU-based critic (`train_semantic_critic.py`) enforces seven MIPS32-specific semantic rules, including stack balance, register usage, branch in-bounds checks, and word alignment. The critic outputs a semantic validity score (`p_valid`) that guides GAN training and filters generated code.

3. **Dataset Refinement:**  
   Generated samples are validated using both an assembler and the semantic critic (`test_GAN_models.py`). Only sequences meeting semantic thresholds (benign τ = 0.104, malware τ = 0.380) are preserved, ensuring that the final dataset is both **syntactically correct** and **semantically valid**.

4. **Hyperdimensional Classification:**  
   The refined dataset is used to train an HDC model (`HDC_Model.py`). Instructions are encoded into bipolar hypervectors, bound with positional hypervectors, and bundled into class representations. Classification is performed via cosine similarity, offering high accuracy with low computational cost.

---

## Contributions
- **Hybrid GAN–HDC Pipeline:** A novel framework that unifies generative modeling with hyperdimensional classification for malware detection.  
- **Semantic Enforcement:** Introduction of a multi-rule semantic critic to enforce program correctness at training time.  
- **Quality-Controlled Dataset Generation:** Production of 27,000 semantically valid benign and malware lines, enabling fair evaluation.  
- **Hardware-Efficient Classification:** Demonstration that HDC achieves strong accuracy while remaining lightweight and suitable for embedded deployment.  
- **Comprehensive Profiling:** Inclusion of training and inference time, CPU/GPU memory usage, and semantic pass rates for reproducibility.

---

## Results
- **Semantic Pass Rate:** Improved significantly when GAN training incorporated critic feedback.  
- **HDC Accuracy:** Achieved competitive accuracy against traditional ML baselines (e.g., Random Forest, SVM), while maintaining drastically lower memory and computational requirements.  
- **Scalability:** The framework demonstrates the feasibility of generating large, high-quality datasets that can generalize across benign and malicious code.  

---

## Acknowledgments
We would like to thank **VirusShare** for providing access to their malware datasets, which were critical in building and evaluating this framework.  