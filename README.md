# MetKG

**MetKG: A knowledge graph-guided framework for interpretable metabolomics**

This repository provides the implementation of **MetKG**, a knowledge graph-guided metabolomics framework designed to identify disease-associated metabolic mechanisms. In this study, MetKG was applied to reveal a kynurenine-related immunometabolic axis in aging and diabetes.

---

## Overview

MetKG integrates metabolomics data with prior biological knowledge represented as a knowledge graph. The framework uses graph-based deep learning to improve disease prediction and provides attention-derived interpretability scores to identify important metabolites, proteins, and metabolite–protein interactions.

The main functions of this repository include:

- training and validating the MetKG model;
- performing prediction using a trained model;
- interpreting important nodes and edges based on model attention scores;
- constructing disease-associated metabolic networks;
- comparing MetKG with baseline models.

---

## Repository Structure

```text
MetKG/
├── model_metkg_train_validation.py      # Model training and validation
├── model_metkg_train_run.py             # Final model training
├── model_metkg_prediction.py            # Prediction using the trained model
├── saved_models/                        # Directory for trained models
├── result/                              # Directory for prediction and analysis results
└── README.md
```

---

## 1. Model Training

The MetKG model can be trained and validated by running:

```bash
python model_metkg_train_validation.py
```

After model validation, the final model can be trained by running:

```bash
python model_metkg_train_run.py
```

The trained model will be saved in the `saved_models/` directory.

---

## 2. Model Prediction

The trained model is saved at:

```text
../saved_models/mtl_graphormer_ple_20251214_122448.pt
```

To perform prediction using the trained model, run:

```bash
python model_metkg_prediction.py
```

The prediction results will be saved in:

```text
../result/
```

---

## 3. Model Interpretability

MetKG provides model interpretability analysis based on attention-derived importance scores. These scores are used to evaluate the importance of nodes and edges in the knowledge graph.

The interpretability analysis can be used to:

- identify important metabolites and proteins;
- evaluate key metabolite–protein interactions;
- construct interpretable disease-associated metabolic networks;
- prioritize candidate biological mechanisms for downstream validation.

---

## 4. Baseline Models

MetKG was compared with multiple baseline models and feature combinations, including:

- Glucose
- Glucose + BMI + Sex
- Random Forest, RF
- Multilayer Perceptron, MLP
- Graph Convolutional Network, GCN
- Graph Attention Network, GAT
- GraphSAGE

These comparisons were used to evaluate the predictive performance and added value of the knowledge graph-guided modeling strategy.

---

## 5. Output

The main output files are saved in:

```text
../result/
```

Typical outputs include:

- prediction results;
- model performance metrics;
- attention-based node importance scores;
- attention-based edge importance scores;
- interpretable metabolite–protein interaction networks;
- baseline model comparison results.

---

## 6. Requirements

The model was developed using Python. The main dependencies include:

```text
python
numpy
pandas
scikit-learn
torch
torch-geometric
networkx
matplotlib
seaborn
```

You may install the required packages using:

```bash
pip install numpy pandas scikit-learn networkx matplotlib seaborn
```

For PyTorch and PyTorch Geometric, please install the versions compatible with your CUDA environment. See:

```text
https://pytorch.org/
https://pytorch-geometric.readthedocs.io/
```

---

## 7. Citation

If you use MetKG in your research, please cite this repository or the related manuscript when available.

```text
MetKG: A knowledge graph-guided framework for interpretable metabolomics
```

---

## 8. Contact

For questions, suggestions, or feedback, please contact:

```text
tianbiao_yang at 126 dot com
```
