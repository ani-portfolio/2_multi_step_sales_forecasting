# <div align="center"> Retail Store Multi-step Sales Forecasting
### Tools & Techniques used:
* Multi-step Time-series forecasting using LightGBM, Linear Regression, Sklearn MultiOutputRegression, RegressorChain
* Components of a Time-Series
    * Trend
    * Seasonality
    * Cyclicality
* Determine components using;
    * Correlogram
    * Periodogram
    * Mutual Information
* Optuna Hyper-parameter Tuning


---
# <div align="center">DESIGN OVERVIEW
This is a multi-step time-series forecasting project focused on predicting item sales in an Ecuadorian grocery store using machine learning. The goal is to make predictions for the next 7 days with a daily inference frequency, using features such as time-dependant features, seasonal features, holidays, among others. 

The requirements are;

- **Prediction Horizon**: 7 Days
- **Lead Time**: 0 Days
- **Prediction Granularity**: Daily
- **Inference Frequency**: Daily

The diagram below is an illustration of the multi-step output required.

![multi_step_output.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/multi_step_output.png)

The project involves exploring time-series components such as trend, seasonality, and cyclicality and using feature engineering and feature selection techniques to build baseline models and multi-step forecasting models.

Four baseline models with be explored;

- **Baseline 1**: Predict last observation for the entire prediction horizon
- **Baseline 2**: Predict Moving average of last 7 days for the entire prediction horizon
- **Baseline 3**: Predict sale in the previous week for the same day in the prediction horizon
- **Baseline 4**: Predict the average of the last 4 weeks for the same day in the prediction horizon

Three types of multi-step forecasting models will be explored;

- **Multi-output model:** These models naturally produce multiple outputs. Linear regression and neural networks are two examples of algorithms that can produce multiple outputs. This strategy is straightforward and efficient, but it may not be possible with every algorithm you want to use. For example, XGBoost is unable to do this.
- **Direct strategy**: Train a separate model for each step in the horizon. For example, one model would forecast 1-step ahead, another model would forecast 2-steps ahead, and so on. Forecasting 1-step ahead is a different problem than forecasting 2-steps ahead, so having a different model make forecasts for each step can be helpful. However, training multiple models can be computationally expensive.
- **Hybrid of Direct & Recursive**: A hybrid of the direct and recursive strategies. It involves training a model for each step and using forecasts from previous steps as new lag features. With each step, each model gets an additional lag input. This strategy can capture serial dependence better than the direct strategy, but it can also suffer from error propagation like the recursive strategy.

Each model will be evaluated against the best baseline model, and amongst each other, to determine which type of model performs best. The models will be evaluated using Mean Absolute Error (MAE).

# <div align="center">DESIGN DOCUMENTATION
## Purpose & Objective

The purpose of this project is to develop multi-step time-series forecasting models to predict item sales in an Ecuadorian grocery store. Grocery stores need item sales predictions for several reasons. First, accurate predictions can help store managers plan inventory levels and ensure that they always have the right amount of stock on hand. This can prevent stockouts, which can lead to lost sales and dissatisfied customers. Additionally, accurate predictions can help store managers make informed decisions about pricing and promotions, which can increase sales and revenue. Finally, item sales predictions can help grocery stores optimize their staffing levels, ensuring that they have enough staff on hand during busy periods and avoiding overstaffing during slow periods. Overall, item sales predictions can help grocery stores operate more efficiently and effectively, leading to increased profitability and customer satisfaction.

The objective is to build a multi-step forecasting machine learning model, that can forecast item sales for the next 7 days, for each item type in a store, for multiple stores. The model must beat a naive baseline model.

## Requirements & Constraints

Requirements and constraints are necessary to ensure that the project meets its objectives and is completed within a specific timeframe. They set clear expectations for what needs to be accomplished and provide a framework for the project plan. By defining the requirements and constraints upfront, the project team can avoid scope creep and ensure that the final product meets the needs of the stakeholders.

**Summary of requirements & constraints:**

- Forecast item sales for each item type in a store, for multiple stores
- Prediction horizon of 7 days
- Lead time of 0 days
- Prediction granularity of daily
- Inference frequency of daily

Since there’s no existing forecasting system to compare performance against, the model performance requirement is to beat a naive baseline model. 

## Methodology

### Problem Statement

The problem statement for this project is to develop multi-step time-series forecasting models to predict item sales in an Ecuadorian grocery store. The goal is to accurately forecast item sales for the next 7 days, at a daily granularity, for each item type in a store, for multiple stores.

Accurate item sales prediction is essential for a grocery store to operate efficiently and effectively. Store managers can use accurate predictions to plan inventory levels and ensure that they always have the right amount of stock on hand. This helps prevent stockouts, which can lead to lost sales and dissatisfied customers. In addition, accurate predictions can help store managers make informed decisions about pricing and promotions, which can increase sales and revenue. Finally, item sales predictions can help grocery stores optimize their staffing levels, ensuring that they have enough staff on hand during busy periods and avoiding overstaffing during slow periods.

The project aims to build a multi-step forecasting machine learning model that can forecast item sales for the next 7 days, at a daily granularity, for each item type in a store, for multiple stores. The model must beat a naive baseline model. The project will involve exploring time-series components such as trend, seasonality, and cyclicality, and using feature engineering and feature selection techniques to build baseline models and multi-step forecasting models.

The success of the project will be measured by the Mean Absolute Error (MAE) of the models. The ultimate goal is to develop a model that can accurately predict item sales, allowing grocery stores to operate more efficiently and effectively, leading to increased profitability and customer satisfaction.

### Data

The dataset is a time-series dataset that includes daily sales data for 33 types of items in 54 stores located in Ecuador. The data spans a period of 4 years and 7 months, from January 1, 2013 to August 15, 2017. The training data consists of sales data from January 1, 2013 to January 1, 2017, while the test data includes sales data from January 1, 2017 to August 15, 2017.

In addition to the sales data, the dataset also includes several exogenous features that can be used for modelling, such as holiday data, daily oil price, daily aggregated transactions at the store level, and store metadata such as city and region. These exogenous features can be used to capture external factors that may impact the sales of the items in the stores.

The time-series dataset without exogenous features looks as follows;

| date | sales | store_family |
| --- | --- | --- |
| 2013-01-01 | 0.0 | 1_AUTOMOTIVE |
| 2013-01-02 | 2.0 | 1_AUTOMOTIVE |
| 2013-01-03 | 3.0 | 1_AUTOMOTIVE |
| 2013-01-04 | 3.0 | 1_AUTOMOTIVE |
| 2013-01-05 | 5.0 | 1_AUTOMOTIVE |
| ... | ... | ... |
| 2017-08-11 | 0.0 | 54_SEAFOOD |
| 2017-08-12 | 1.0 | 54_SEAFOOD |
| 2017-08-13 | 2.0 | 54_SEAFOOD |
| 2017-08-14 | 0.0 | 54_SEAFOOD |
| 2017-08-15 | 3.0 | 54_SEAFOOD |

The initial dataset was not a continuous time-series because there were some missing dates. However, after cross-checking the missing dates with holidays, it was clear that the missing dates were on holidays, where sales were 0. These missing dates were added into the dataset with 0 sales, to make it a continuous time-series. 

### Baseline Models

Four baseline models will be explored. They are;

- **Baseline 1**: Baseline 1 is a forecasting model that predicts that the item sales for the entire prediction horizon will be the same as the last observed value. For example, if the last observed value was 100, the model will predict that the item sales for the entire prediction horizon will be 100.
- **Baseline 2**:  Baseline 2 model predicts the moving average of the last 7 days for the entire prediction horizon. This means that the model takes the average of the sales for the previous 7 days and uses that as the prediction for each day in the prediction horizon.
- **Baseline 3**: Baseline 3 predicts the sales of an item for a given day in the prediction horizon, using the sales of the same item for the same day of the week in the previous week. For example, to predict the sales of an item for Monday of the next week, this baseline model will use the sales of that item for Monday of the previous week. This baseline model is based on the assumption that sales for a given day of the week tend to be similar across different weeks, and thus can be used as a predictor for the same day in the next week.
- **Baseline 4**: Baseline 4 predicts the average sales for the same day in the prediction horizon, based on the average sales of the same day of the week over the past four weeks. For example, to predict the sales of an item for Monday of the next week, this baseline model will use the average sales for that item for the past four Mondays. The idea behind this model is that sales tend to be similar for a given day of the week, so using the average sales for the same day of the week over the past four weeks can be a good predictor for the same day in the next week.

### Techniques

**Correlogram**

A correlogram is a plot that shows the correlation between a time series and its lagged values, up to a certain number of lags. It can help identify the appropriate lag values to use in a time-series model. The partial autocorrelation plot is a related plot that shows the correlation between a time series and its lagged values while controlling for the values of other lags. The correlogram for average sales of all stores and items is shown below. It is also called a Partial Autocorrelation plot.

![correlogram.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/correlogram.png)

Lags 1 to 7, 14, and 28 seem to have high correlation with the target. 

**Periodogram**

The periodogram is a tool used to identify the frequency components of a time series. It is particularly useful for identifying the frequencies that contribute to seasonal patterns in the data. By analyzing the periodogram, the number of Fourier features to use in a model can be determined. The periodogram for average sales of all stores and items is shown below.

![periodogram.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/periodogram.png)

Based on the periodogram, Fourier features up to 104 seem to be appropriate. However, Weekly and Semiweekly patterns are captured using date-time features, and therefore don’t need to be captured using Fourier features. Therefore, a lower number of Fourier features, can be used. 

**Mutual Information**

Mutual Information is a metric used to measure the amount of information shared by two variables. In the context of time-series forecasting, it can be used to determine the relationship between lag features and the target variable. Unlike correlation, mutual information can detect any kind of relationship between variables, not just linear relationships. However, mutual information is a univariate metric, so it cannot detect interactions between features. 

Three multi-step forecasting models will be explored as described below;

1. **Multi-output model:** These models naturally produce multiple outputs. Linear regression and neural networks are two examples of algorithms that can produce multiple outputs. This strategy is straightforward and efficient, but it may not be possible with every algorithm you want to use. For example, XGBoost is unable to do this.
2. **Direct strategy:** Train a separate model for each step in the horizon. For example, one model would forecast 1-step ahead, another model would forecast 2-steps ahead, and so on. Forecasting 1-step ahead is a different problem than forecasting 2-steps ahead, so having a different model make forecasts for each step can be helpful. However, training multiple models can be computationally expensive.
3. **Hybrid of Direct & Recursive:** A hybrid of the direct and recursive strategies. It involves training a model for each step and using forecasts from previous steps as new lag features. With each step, each model gets an additional lag input. This strategy can capture serial dependence better than the direct strategy, but it can also suffer from error propagation like the recursive strategy.

**Bayesian Hyper-parameter Tuning**

Optuna is a Python package that provides a framework for hyper-parameter tuning. It uses a Bayesian optimization algorithm that adapts to the results of previous trials in order to find the best set of hyper-parameters for a given machine learning model. This allows for a more efficient and effective approach to hyper-parameter tuning, as it reduces the number of trials required to find the optimal hyper-parameters. In this project, Optuna was used to optimize the performance of the LightGBM algorithm, resulting in improved accuracy for the NYC taxi demand prediction model.

Since this is a time-series model, it is important to respect the temporal order of the data during hyper-parameter tuning. I.e. Ensure the temporal order of the data is intact when the training data is split. This can be achieved easily using Sklearn TimeSeriesSplit. 

**Feature Engineering**

Feature engineering was performed with the objective of capturing the 3 components of a time-series, namely;

- **Trend**: Some store items show a trend in sales as seen in the figure below. The red line represents a 365 day moving average, that increases over time.
    
    ![trend.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/trend.png)
    

- **Seasonality:** When sales of all stores and items are averaged, and plotted by day of the week, a clear weekly pattern can be observed.
    
    ![weekly_sale.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/weekly_sale.png)
    

- **Cyclicality:** Cyclicality can be modelled using lag features based on the correlogram. When sales of all stores and items are averaged, lag features between 1 - 7, and 14, and 28 seem to have high correlation with the target.
    
    ![correlogram.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/correlogram.png)
    

In addition to exogenous features such as holidays and daily oil price, the following features were included to capture trend, seasonality and cyclicality.

- An increasing value over time, to capture trend
- Add date-time features such as day of week, to capture seasonality
- Add Fourier Features based on Periodogram, to capture seasonality
- Add average of the sales for the same day over the past four weeks, to capture seasonality
- Add Lag features based on Correlogram, to capture cyclicality

### Model Workflow Diagram

Since this model is not in production, and is only a PoC, a full architecture diagram is not available. The following is a diagram that outlines the steps required to create the model, at a high level. 

![model_workflow.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/model_workflow.png)

### Evaluation

**Mean Absolute Error**

MAE stands for Mean Absolute Error. It is a metric used to evaluate the performance of a regression model. The metric is calculated by taking the average of the absolute differences between the predicted values and the actual values. A lower MAE indicates that the model is better at predicting the target variable. 

The formula for MAE is;

$$ \frac{1}{n}\sum_{i=1}^{n}|x_i-y_i| $$

Where $n$ is the number of samples, $x$ is the prediction and $y$ is the actual value. 

The MAE can be used to evaluate the performance of the baseline models, and how the LightGBM model performs against simple baseline models.


# <div align="center">RESULTS
The results of the 4 baseline models, and 3 multi-step forecasting models are tabulated below.

| Model | Description | MAE | Percent Difference |
| --- | --- | --- | --- |
| Baseline 1 | Predict last observation for the entire prediction horizon | 151.3518 |  |
| Baseline 2 | Predict Moving average of last 7 days for the entire prediction horizon | 118.5240 | -21.69% |
| Baseline 3 | Predict sale in the previous week for the same day in the prediction horizon | 96.0924 | -18.93% |
| Baseline 4 | Predict the average of the last 4 weeks for the same day in the prediction horizon | 84.1223 | -12.46% |
| Multi-Step Model 1 | Multi-Output model using Linear Regression | 86.3212 | 2.61% |
| Multi-Step Model 2 | Direct Multi-step model using LightGBM and Sklearn MultiOutputRegressor | 74.5421 | -13.65% |
| Multi-Step Model 3 | Direct-recursive hybrid model using LightGBM and SKlearn RegressorChain | 73.4028 | -1.53% |

Baseline model 4 had the best MAE score, and the Direct-recursive hybrid model had the best MAE score overall. 

The prediction plots for each model are shown below. The plots are for 1 specific store-item pair. The different colours represent a single inference (7 steps forward) on a given day. 


********************************Baseline Model 1:******************************** MAE 151.3518

![baseline_1.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/baseline_1.png)
---

********************************Baseline Model 2:******************************** MAE 118.5240

![baseline_2.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/baseline_2.png)
---

********************************Baseline Model 3:******************************** MAE 96.0924

![baseline_3.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/baseline_3.png)
---

********************************Baseline Model 4:******************************** MAE 84.1223

![baseline_4.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/baseline_4.png)
---

********************************Multi-Output Model:******************************** MAE 86.3212

![multi_output.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/multi_output.png)
---

********************************Direct Multi-step Model:******************************** MAE 74.5421

![direct_multi_step.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/direct_multi_step.png)
---

********************************Direct-Recursive Hybrid Model:******************************** MAE 73.4028

![direct_recursive.png](https://github.com/ani-portfolio/2_retail_store_sales_forecasting/blob/3_model_building/docs/direct_recursive.png)
---
