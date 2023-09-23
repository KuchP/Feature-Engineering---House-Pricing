# Feature-Engineering---House-Pricing

# House Price Prediction Analysis

## Techniques Employed:

### Data Cleaning and Exploration:

- **Handling Missing Values**: Identified columns with missing values and implemented strategies to impute or remove them based on the nature of the variable.
- **Outlier Detection and Treatment**: Used visualization techniques, such as box plots, to detect outliers and applied methods like IQR to treat them.
- **Exploratory Data Analysis (EDA)**: Employed various visualization techniques, including histograms, scatter plots, and correlation matrices, to gain insights into individual features and their relationships.

### Feature Engineering:

#### Numerical Data:
- **Skewness Treatment**: Applied log transformation to positively skewed features to normalize their distribution.
- **Scaling**: Standardized numerical features to have a mean of zero and a variance of one using the `StandardScaler`.

#### Categorical Data:
- **Category Aggregation**: Bundled low-frequency categories into a singular 'Other' category to prevent sparse data issues.
- **Encoding**: Used one-hot encoding to transform categorical variables into a machine-readable format.

#### Date Features:
- **Extraction**: Extracted year and month components from date columns to capture any time trends or seasonality in the data.

- **Interaction Terms**: Generated interaction terms between features to capture combined effects, which might be significant for house price prediction.
- **Polynomial Features**: Introduced polynomial terms for certain features to capture non-linear relationships.

### Feature Selection:
- **Recursive Feature Elimination (RFE)**: Employed RFE with cross-validation to rank features based on their significance in predicting house prices. This technique systematically trimmed down the feature set to the most influential predictors.

### Model Building:
- **GradientBoostingRegressor**: Chosen for its ability to capture non-linear relationships and interactions between features.
- **Pipeline Integration**: Incorporated the preprocessing steps and model training within a pipeline. This ensured the consistency of data transformations across training and test datasets.
- **Hyperparameter Tuning with Bagging**: Used a combination of `BaggingRegressor` and `RandomizedSearchCV` within the pipeline. This allowed for a robust search over specified hyperparameter values of the estimator, leading to an optimized model with the best parameters.

### Model Evaluation:
- **R2 Score**: Utilized the R2 score metric to assess the proportion of variance in the target variable that is predictable from the feature variables.
- **Cross-validation**: Applied k-fold cross-validation to get a more comprehensive understanding of the model's performance across different subsets of the data.
- **Hyperparameter Insights**: Assessed the best parameters returned by the `RandomizedSearchCV` to understand the optimal configurations for the model.

### Deeper Estimation Post Feature and Parameter Selection:
After the initial modeling, a more profound estimation was carried out, employing advanced techniques and methodologies. This deep dive aimed to refine the predictions further, ensuring that the model's outputs were not only accurate but also reliable across various scenarios.

## RESULTS:

### Model Performance on Training Data:

- **Gradient Boosting Trees (GBT)** achieved the highest R2 score of approximately 0.9737 on the training data, suggesting that the model was able to capture the patterns in the training data very well.
- **Decision Trees (DT)** and **Random Forests (RF)** also performed exceptionally well with R2 scores of around 0.9789 and 0.9491 respectively.
- **Support Vector Regression (SVR)** and **K-Nearest Neighbors (KNN)** had moderate performance with R2 scores of 0.5317 and 0.8675, respectively.
- **Linear Regression (LR)** had a score of 0.7859, which is decent but lower in comparison to tree-based models.

### Model Performance on Test Data:
- **GBT** remained the top performer with an R2 score of approximately 0.8766 on the test data. This indicates that the model generalizes well to unseen data.
- **DT** and **RF** followed closely with scores of 0.8484 and 0.8642 respectively.
- **SVR** improved slightly on the test data with a score of 0.5185, but it's still considerably lower than the tree-based models.
- **KNN** and **LR** had scores of 0.8359 and 0.7591 respectively, showing decent generalization abilities.

### Overall Insights:
- Tree-based models, especially **Gradient Boosting Trees**, demonstrated superior performance both on training and test datasets, making them the preferred choice for this specific problem.
- Despite the high training scores of **Decision Trees** and **Random Forests**, their test scores are slightly lower than the **GBT**, indicating a potential overfitting scenario.
- The **SVR** model's performance was notably lower than other models, suggesting that the data might not adhere to the assumptions or patterns that **SVR** typically captures.
- The **KNN** model's performance was moderate, but given its nature, it might require further feature engineering or scaling for optimal performance.
- **Linear Regression**, being a simple model, provided a decent baseline, but more complex models like **GBT** outperformed it.

## Recommendations:

- **Gradient Boosting Trees** should be the primary choice for deployment given its robustness and high performance on test data.
- Regular monitoring and validation should be conducted to ensure the model's performance remains consistent over time.
