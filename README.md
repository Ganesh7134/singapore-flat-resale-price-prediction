# singapore-flat-resale-price-prediction

* **Brief description**: A project to develop a model for predicting resale prices of flats in Singapore, aiming to provide insights for homeowners, investors, and policymakers.
* **Key questions**: What factors significantly influence flat resale prices? Can we accurately predict prices for informed decision-making?
* **Target audience**: Homebuyers, sellers, property agents, investors, government agencies, researchers.
  
## Data

* **Sources**: Data.gov.sg, Urban Redevelopment Authority (URA), and other relevant sources.
* **Preprocessing**: Cleaning, handling missing values, feature engineering (e.g., creating new features for location attributes).
* **Format and storage**: CSV format, stored in a designated folder within the project directory.
  
## Model

* **Modeling approach**: XGBoost (chosen for its performance in similar tasks).
* **Evaluation metrics**: R-squared, MAE, RMSE ,MSE
* **Hyperparameter tuning**: Grid search to optimize model performance.
* **Feature importance analysis**: Identifying key factors driving prices.
  
## Usage

* **Environment setup**: Prerequisite libraries (e.g., pandas, XGBoost) listed in requirements.txt.
* **Running code**: Instructions on executing model training and prediction scripts.
* **Making predictions**: Examples of using the trained model to predict prices for new flat listings.
  
## Results

* **Key findings**: Highlight model performance metrics and key insights from feature importance analysis.
* **Visualizations**: Plots of model results and feature importance.
* **Limitations**: Discuss model constraints and potential areas for improvement.

## Conclusion

This project successfully developed a model for predicting resale prices of flats in Singapore, achieving an R-squared of 0.683, MAE of 0.11, and RMSE of 0.12. While these scores indicate a moderate ability to capture price trends, there's room for improvement.
