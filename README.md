# Titanic-Disaster-Analysis
## Project Overview
This project focuses on exploring and cleaning the Titanic dataset, which is a well-known dataset in the machine learning and data science community. The goal of this project is to perform thorough data cleaning, preprocessing, and transformation tasks to prepare the dataset for analysis or predictive modeling. The dataset contains information about passengers aboard the Titanic, including attributes such as their age, sex, class, and whether they survived or not. The primary objectives are to identify missing data, handle outliers, encode categorical variables, and summarize the cleaned dataset for analysis.

![Titanic Logo](https://github.com/HimanshuSharma123-a/Titanic-Disaster-Analysis/blob/main/Titanic.webp)

# Titanic Dataset - Data Cleaning and Analysis

## Dataset Information
The dataset used in this project is the Titanic dataset, available on [Kaggle](https://www.kaggle.com/c/titanic/data). This dataset includes the following columns:
- **PassengerId**: A unique identifier for each passenger.
- **Pclass**: The class of the passenger (1st, 2nd, or 3rd).
- **Name**: The name of the passenger.
- **Sex**: The gender of the passenger.
- **Age**: The age of the passenger.
- **SibSp**: The number of siblings/spouses aboard the Titanic.
- **Parch**: The number of parents/children aboard the Titanic.
- **Ticket**: The ticket number.
- **Fare**: The fare paid for the ticket.
- **Cabin**: The cabin where the passenger stayed.
- **Embarked**: The port where the passenger boarded the Titanic (C = Cherbourg; Q = Queenstown; S = Southampton).
- **Survived**: Whether the passenger survived (0 = No, 1 = Yes).

---

## Steps Involved in the Project

### 1. **Data Loading**
The dataset is loaded using Pandas for further exploration and manipulation.

```python
import pandas as pd
data = pd.read_csv('titanic.csv')

