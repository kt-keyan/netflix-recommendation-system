# 🎬 Netflix Movie Recommendation System

A personalised movie recommendation engine built using **SVD (Singular Value Decomposition)** matrix factorisation on the real **Netflix Prize dataset**. The system processes over 100,000 user ratings, filters noise intelligently, and generates ranked movie recommendations with predicted scores for any user.

---

## 📌 Problem Statement

With thousands of movies available, users often struggle to find content they'll enjoy. This project tackles that by learning from historical rating patterns to predict how much a user would like a movie they haven't seen — and recommending the top ones.

---

## 📂 Dataset

- **Source**: [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) (Kaggle)
- `combined_data_1.txt` — Raw ratings file (Movie IDs embedded as row markers)
- `movie_titles.csv` — Movie ID to title/year mapping

---

## ⚙️ Workflow

### 1. Data Loading & Parsing
- Loaded raw Netflix Prize format where Movie IDs appear as `NaN` rows
- Built a custom mapping pipeline to assign correct Movie IDs to each rating row

### 2. Exploratory Data Analysis
- Analysed rating distribution across 1–5 stars
- Counted total movies, unique customers, and total ratings in the dataset

### 3. Noise Filtering
- Removed movies with fewer reviews than the **70th percentile** of movie review counts
- Removed customers with fewer reviews than the **70th percentile** of customer review counts
- This ensures the model trains on active users and popular movies, improving signal quality

### 4. Model Building — SVD
- Used the **Surprise** library's SVD algorithm (matrix factorisation)
- Trained on 100,000 ratings sampled from the cleaned dataset
- Evaluated using **3-fold cross-validation** with RMSE and MAE metrics

### 5. Generating Recommendations
- Fitted SVD on the full training set
- For a target user, predicted ratings for all movies they haven't seen
- Sorted predictions by estimated score to produce a ranked recommendation list

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading, cleaning, filtering |
| NumPy | Array operations |
| Matplotlib | Rating distribution visualisation |
| Surprise (scikit-surprise) | SVD model & cross-validation |

---

## 📊 Results

- Model evaluated via **3-fold cross-validation**
- Metrics: **RMSE** and **MAE** on held-out ratings
- Output: Ranked list of recommended movies with predicted rating scores per user

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/kt-keyan/netflix-recommendation-system.git
cd netflix-recommendation-system

# 2. Install dependencies
pip install pandas numpy matplotlib scikit-surprise

# 3. Add the dataset files to the project folder
# - combined_data_1.txt
# - movie_titles.csv
# (Download from: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

# 4. Open and run the notebook
jupyter notebook netflix_project.ipynb
```

---

## 📁 Project Structure

```
netflix-recommendation-system/
│
├── netflix_project.ipynb   # Main notebook
├── movie_titles.csv        # Movie ID → Title mapping
├── combined_data_1.txt     # Netflix Prize ratings (add manually)
└── README.md
```

> **Note**: `combined_data_1.txt` is not included in this repo due to file size. Download it from Kaggle using the link above.

---

## 👤 Author

**Karthikeyan K**  
B.Tech — AI & Data Science, Annai Mira College of Engineering and Technology  
📧 karthikeyank00118@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/karthi-keyan-3b44a32a8/) | [GitHub](https://github.com/kt-keyan)
