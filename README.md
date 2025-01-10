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

2. Initial Exploration

A basic inspection of the data is performed to understand its structure and identify any potential issues such as missing values or incorrect data types.

data.head()
data.info()

3. Handling Missing Values

Missing values in columns such as Age and Embarked are identified. The missing values are then filled with appropriate strategies (e.g., filling numerical columns with the mean and categorical columns with the mode).

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

4. Removing Duplicates

Duplicate rows are checked and removed to ensure data integrity.

data = data.drop_duplicates()

5. Feature Engineering

New features are created, such as FamilySize, which combines SibSp and Parch to represent the total number of family members traveling with the passenger.

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

6. Handling Categorical Data

Categorical columns like Sex and Embarked are encoded numerically for easier analysis and model compatibility.

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

7. Outlier Detection and Removal

Outliers in the Fare column are identified using the Interquartile Range (IQR) method and removed to avoid skewed analysis.

Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Fare'] >= lower_bound) & (data['Fare'] <= upper_bound)]

8. Data Scaling

Some features, like Age and Fare, are normalized or scaled to ensure that they are on a comparable scale.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

9. Final Data Check

After performing all cleaning steps, a final check is made to ensure that there are no remaining missing values or issues.

data.isnull().sum()
data.info()

##This project demonstrates the importance of data cleaning in preparing datasets for analysis. Data cleaning involves multiple steps such as handling missing values, removing duplicates, encoding categorical variables, and transforming features, all of which contribute to ensuring the dataset is in a usable form for machine learning models.


