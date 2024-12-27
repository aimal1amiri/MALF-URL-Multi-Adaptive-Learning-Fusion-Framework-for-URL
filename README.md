<h1 align="center">MALF-URL: Multi-Adaptive Learning Fusion<br>Framework for URL Classification with Dynamic<br>Feature Orchestration</h1>

###

<h4 align="left">Abstract</h4>

###

<p align="left">MALF-URL is developed to tackle the growing sophistication of phishing and malicious URLs. The system leverages a unique combination of BERT-based natural language processing and traditional machine learning models to improve URL classification. Through a dynamic feature fusion mechanism, the framework integrates semantic and structural patterns, achieving robust adaptability across diverse URL types. A meta-learning framework combines eight classifiers, including Random Forest, XGBoost, LightGBM, and SVM, to deliver superior accuracy. MALF-URL sets a benchmark in cybersecurity by effectively identifying even the most elusive phishing attempts.</p>

###

<h2 align="left"></h2>

###

<h5 align="left">Features</h5>

###

<p align="left">Model: Combines deep learning (BERT) with traditional machine learning classifiers.<br>Dynamic Feature Fusion: Balances the importance of semantic and structural URL features.<br>Adaptive Meta-Learner: Utilizes dynamic weighting to optimize classifier contributions.<br>Comprehensive Feature Analysis: Extracts structural features like URL length, path length, domain characteristics, and combines them with BERT-derived embeddings.<br>Scalability: Designed to adapt to evolving phishing threats with robust generalization across diverse datasets.<br>Performance Benchmarks: Achieves high accuracy, precision, and recall metrics, outperforming standalone approaches.</p>

###

<h2 align="left"></h2>

###

<h2 align="left"></h2>

###

<h5 align="left">Methodology</h5>

###

<p align="left">Steps:<br><br>    System Architecture:<br><br>    Combines BERT-based embedding for semantic analysis with traditional URL feature extraction.<br>    Features are processed through a multi-model ensemble framework.<br><br>BERT-Based Embedding:<br><br>    Uses DistilBERT for URL tokenization and transformation into 768-dimensional vectors.<br>    Incorporates dropout regularization and activation functions to optimize feature representation.<br><br>Dynamic Feature Extraction:<br><br>    Processes structural characteristics such as HTTPS status, URL length, and special character counts.<br>    Normalizes features using StandardScaler and augments data for robustness.<br><br>Ensemble Framework:<br><br>    Integrates predictions from eight classifiers (e.g., Random Forest, XGBoost, SVM) using logistic regression as a meta-learner.<br>    Dynamically adjusts classifier contributions based on real-time performance.<br><br>Adaptive Fusion Mechanism:<br><br>    Implements a learnable parameter to balance BERT embeddings and structural features for each URL.</p>

###

<h2 align="left"></h2>

###

<h5 align="left">Results</h5>

###

<p align="left">Performance Metrics:<br><br>    Accuracy: Achieved up to 96.2% accuracy in URL classification tasks.<br>    F1-Score: Consistently high F1-scores across validation folds, with a maximum of 0.9875 for BERT alone.<br>    False Positives: Reduced false positives by 18% compared to static ensemble approaches.<br>    Confusion Matrix: Demonstrated near-perfect classification, with 97.3% accuracy for legitimate URLs and zero false negatives for phishing URLs.</p>

###

<h2 align="left"></h2>

###

<h2 align="left">code with</h2>

###

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" height="40" alt="tensorflow logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" height="40" alt="jupyter logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40" alt="numpy logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="40" alt="matlab logo"  />
</div>

###
