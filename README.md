Student Placement Prediction System
Project Overview:

This project focuses on predicting whether a student is likely to get placed in a job based on different academic and skill-related factors. The idea behind this project is to analyze how features such as GPA, coding ability, internships, and technical skills influence a student's chances of getting placed.
To achieve this, several machine learning models were applied and compared. The final model is integrated into a Streamlit web application, where users can enter student details and instantly receive a placement prediction along with the probability of being placed.

Dataset:
The dataset used in this project contains information about students and their academic as well as skill-based attributes.

Some of the important features include:

College GPA
Coding Score
Internships
Projects
Technical Skills
Communication Skills
Certifications
Hackathons
Leadership and extracurricular activities

The target variable in the dataset is placed, which indicates whether the student was placed (1) or not placed (0).

Exploratory Data Analysis (EDA):

Before building the machine learning models, exploratory data analysis was performed to better understand the dataset.

During EDA, the following steps were carried out:

Checking for missing values and duplicates
Understanding the distribution of different features
Identifying relationships between variables
Analyzing how different features affect placement
Various visualizations such as histograms, boxplots, pairplots, and correlation heatmaps were used to identify patterns in the data.
Some key observations from the analysis were:
Students with higher coding scores tend to have better placement chances.
Internships and projects play an important role in improving placement probability.
Technical and communication skills also contribute significantly to placement outcomes.

Machine Learning Models:

To predict student placement, multiple machine learning models were implemented and compared.

The models used in this project include:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)

Each model was evaluated using performance metrics such as accuracy, precision, recall, and F1-score.
Among all the models tested, Random Forest performed the best, as it was able to capture complex relationships between the features while maintaining good prediction accuracy.

Streamlit Web Application:

To make the model easier to use, a simple Streamlit web interface was created. This allows users to interact with the model without needing to write any code.

Through this interface, users can enter the following information:

College GPA
Coding Score
Number of Internships
Number of Projects
Technical Skills Score
Communication Skills Score

After entering the details, the system predicts whether the student is likely to be placed and also displays the probability of placement.

Technologies Used:

The project was developed using the following tools and libraries:

Python – Programming language
Pandas & NumPy – Data handling and preprocessing
Matplotlib & Seaborn – Data visualization
Scikit-learn – Machine learning models
Joblib – Model saving and loading
Streamlit – Building the web interface

How to Run the Project:

Install the required libraries:

pip install streamlit pandas numpy scikit-learn joblib

Run the Streamlit application:

streamlit run app.py

Open the application in your browser at:

http://localhost:8501

Future Improvements:

Some improvements that can be added to this project include:

Using more advanced machine learning models
Improving feature engineering
Adding a dashboard with more visual insights
Deploying the application online
Allowing batch predictions using uploaded datasets
