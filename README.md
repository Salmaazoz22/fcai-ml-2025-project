# Machine Learning Projects

This repository contains two machine learning projects:  

1. **Facial Emotion Recognition** – Supervised and Unsupervised learning on facial images.  
2. **Bank Loan Approval Prediction** – Predicting loan approval using classification models.

---

## 1️⃣ Facial Emotion Recognition

### 📂 Dataset
- **Source:** Adapted CK+ Facial Expression Dataset (up to 920 images)  
- **Image Details:** 48×48 pixels, grayscale, face-cropped using `haarcascade_frontalface_default`  
- **Preprocessing:** Adjusted to reduce lighting, hair, and skin-color noise  
- **Columns:** `emotion`, `pixels`, `Usage`  
- **Reference:** [FER2013 by Rohit Verma](https://www.kaggle.com/datasets/deadskull7/fer2013)  

**Emotion Classes:**  

| Label | Emotion   | Samples |
|-------|----------|---------|
| 0     | Anger    | 45      |
| 1     | Disgust  | 59      |
| 2     | Fear     | 25      |
| 3     | Happiness| 69      |
| 4     | Sadness  | 28      |
| 5     | Surprise | 83      |
| 6     | Neutral  | 593     |
| 7     | Contempt | 18      |

### 🔹 Models & Features
| Model | Learning Type | Features | Accuracy |
|-------|---------------|---------|----------|
| Logistic Regression | Supervised | Raw Pixels | 95.83% |
| K-Means + HOG      | Unsupervised | HOG Features | 74.15% |

- Logistic Regression: Confusion matrix, ROC curve, and loss curve evaluation  
- K-Means: t-SNE visualization, cluster-to-label mapping  

---

## 2️⃣ Bank Loan Approval Prediction

### 📂 Dataset
- **Source:** Bank Loan Dataset (~5000 records, 14 columns)  
- **Features:** Demographics, financial info, credit history, loan details  
- **Target:** Loan approval (`Yes` / `No`)  

### 🔹 Models
| Model | Learning Type | Accuracy |
|-------|---------------|---------|
| Linear Regression | Supervised | 87% |
| K-Nearest Neighbors (KNN) | Supervised | ~81% |

- Data preprocessing: Handling missing values, encoding categorical variables, scaling numerical features  
- Evaluation: Confusion matrix, accuracy score, and classification report  

---

## 📈 Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/ml-projects.git
cd ml-projects
