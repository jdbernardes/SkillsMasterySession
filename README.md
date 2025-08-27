# SkillsMasterySession: 📊 Impact of Imbalanced Structured Data on AI Model Performance

## 🧠 Project Purpose

This project demonstrates how **imbalanced structured data** can negatively affect the performance and fairness of AI models. Using the **UCI Adult Income Dataset**, we simulate bias in model predictions caused by underrepresentation of certain groups (e.g., gender), and show how rebalancing the data can improve outcomes.

---

## 📁 Dataset Overview

- **Source**: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)
- **Description**: Predict whether an individual earns more than \$50K/year based on demographic and employment attributes.
- **Features**:
  - Age, Workclass, Education, Marital Status, Occupation, Race, Sex, Hours-per-week, etc.
- **Target**: Income (<=50K or >50K)

---

## ⚠️ Problem Addressed

AI models trained on **imbalanced data** can develop biased behaviors. For example:
- If female entries are underrepresented, the model may underpredict high income for women.
- This leads to **unfair outcomes** and **reduced model reliability**.

---

## 🧪 Demo Steps

1. **Load the Full Dataset**
   - Train a baseline model using balanced data.
   - Evaluate performance across demographic groups.

2. **Create an Imbalanced Dataset**
   - Undersample female entries to simulate bias.
   - Retrain the model and observe skewed predictions.

3. **Rebalance the Dataset**
   - Apply oversampling or class weighting techniques.
   - Retrain and compare fairness and accuracy.

---

## ✅ Expected Outcomes

- Visual evidence of how imbalance affects predictions.
- Improved fairness and performance after rebalancing.
- Increased awareness of the importance of **data quality and representation** in AI development.

---

## 🧑‍💼 Relevance to SAP

Structured data is common in enterprise systems. This demo reflects real-world scenarios where biased training data can impact:
- HR analytics
- Financial predictions
- Customer segmentation

By understanding and correcting data imbalance, teams can build **more ethical and effective AI solutions**.