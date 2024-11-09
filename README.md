# Wildfire project Model 
This project focuses on predicting wildfire occurrences using historical data and machine learning techniques. The model incorporates classification and regression algorithms to forecast the likelihood of wildfire events and assess related risk factors such as fuel moisture and duff moisture codes.

Classification and Regression Algorithms
For classification, we employed the Gradient Boosting Classifier due to its ability to handle complex data relationships and achieve high accuracy, with a final model accuracy of 90%. This model outperformed others in precision and recall, making it suitable for accurate wildfire prediction. Other classification models considered included Random Forest and Logistic Regression, but Gradient Boosting showed the best balance of accuracy and computational efficiency.

For regression, we used the Gradient Boosting Regressor to estimate continuous variables, achieving an R² score of 0.92 and a Mean Squared Error (MSE) of 2.87. This choice was based on its ability to model non-linear relationships in the data, making it effective for continuous variable predictions such as fire spread potential.

Data and Parameters
Key environmental factors utilized in the model include:

Temperature
Wind Speed
Fuel Moisture Code (FMC)
Duff Moisture Code (DMC)
Initial Spread Index (ISI)
These parameters are critical as they influence fire ignition and behavior, and are sourced from national weather stations and wildfire incident reports. After preprocessing, the data was normalized and split into training, validation, and test sets.

Model Evaluation
To evaluate classification performance, we used accuracy, precision, recall, and F1-score, with accuracy being the primary metric. For regression, we employed Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² to measure model fit. These metrics helped fine-tune the models, ensuring the best-performing ones were selected for deployment.

By using a robust combination of Gradient Boosting techniques, this project provides an effective tool for proactive wildfire risk management and enhances decision-making for resource allocation and early warning systems.







