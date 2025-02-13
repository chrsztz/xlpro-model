# 🎹 Deep Learning-Based Piano Fingering Prediction Model 🎶  
<p align="center">
  🌍 <a href="https://github.com/chrsztz/xlpro-model/blob/main/readme_en.md">English Version</a> | 🎹 <a href="https://github.com/chrsztz/xlpro-model">中文版本</a> 🌍
</p>

In collaboration with the **PhD research lab at our school** and a **PhD candidate in Computer Science at the National University of Defense Technology**, this project focuses on **AI deep learning and cloud computing**, aiming to develop an **AI-powered piano fingering prediction model**.  

---

## 🔬 Research Objective  

The goal of this study is to leverage deep learning techniques to **predict piano fingering from sheet music**, while optimizing model accuracy and generalization capabilities.  

---

## 📊 Data Processing  

1. **Dataset Selection**: We use the **PIG dataset** for sequential piano sheet music processing.  
2. **Data Preprocessing**:  
   - **Normalization and mapping of pitch names**  
   - Extraction of **multi-dimensional features**, including:  
     - 🎵 **Playing speed**  
     - 🎼 **Note density**  
     - 🎶 **Chord structure**  
     - ⏳ **Note duration**  
     - 🔢 **Pitch information**  
3. **Data Augmentation**:  
   - **SMOTE oversampling** to balance the dataset  
   - **Symmetric mirroring augmentation** to enhance generalization  

---

## 🧠 Model Architecture  

1. **Feature Fusion**:  
   - **Word2Vec** for feature vectorization  
   - **Standardization of numerical features** to improve model stability  
2. **Deep Learning Framework**:  
   - **BiLSTM (Bidirectional Long Short-Term Memory)** to capture **sequential dependencies in note progression**  
   - **Attention mechanism** to enhance **weight distribution for key features**  
   - **CRF (Conditional Random Field)** for **global optimization of fingering sequence labeling**  

---

## 🎯 Training & Testing Results  

✅ **Model Accuracy**: **89%**  
📉 **Loss Rate**: **0.63**  

The model has been successfully **deployed on a cloud server** and is currently in the **testing phase**.  
Further optimization of the **algorithm framework** is needed to enhance model performance and prediction accuracy.  

---

## 🚀 Future Improvements  

- Further **refinement of data augmentation strategies**  
- Exploring **more sophisticated sequence modeling techniques** for better generalization  
- **Expanding the dataset** by incorporating the **ThumbSet dataset** to improve diversity  

---

💡 **The research is ongoing, and we look forward to achieving even better results!** 🚀🎶  

---
