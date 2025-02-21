# Time Series Exploration: Libraries and Datasets

This repository documents a session exploring various time series libraries and datasets in Python. The goal is to gain hands-up experience with different approaches to time series analysis, forecasting, and anomaly detection.

**Focus of this Session:**

*   Hands-on experimentation with different time series libraries.
*   Exploring various datasets with diverse characteristics (seasonality, trends, noise levels, etc.).
*   Comparing the functionalities and ease of use of different libraries.
*   Building a basic understanding of common time series tasks (e.g., forecasting, anomaly detection).

**Libraries Explored (Including Target Libraries):**

*   **Darts:** [https://unit8co.github.io/darts/](https://unit8co.github.io/darts/) -  A Python library for easy manipulation and forecasting of time series.  It provides a unified interface to a variety of models, including classical statistical models (e.g., ARIMA, Exponential Smoothing), machine learning models (e.g., Regression models, Random Forests), and deep learning models (e.g., N-BEATS, Transformer). Darts also supports multivariate time series, probabilistic forecasting, and backtesting.  This is a primary focus of this session.
*   **Statsmodels:** A well-established Python library for statistical modeling, including time series analysis with classical models like ARIMA, SARIMA, VAR, etc.
*   **Prophet:** Developed by Facebook, Prophet is designed for forecasting time series data with strong seasonality and trend components.  It's particularly good at handling missing data and outliers.
*   **scikit-learn:** While not exclusively a time series library, scikit-learn provides many useful tools for time series analysis, such as regression models, feature engineering techniques, and cross-validation methods that can be applied to time series problems.
*   **PyTorch Forecasting:** A PyTorch-based library specifically for time series forecasting with neural networks. It provides implementations of state-of-the-art models and utilities for data loading, preprocessing, and evaluation.
*   **TensorFlow Probability:**  Provides tools for probabilistic modeling and inference, which can be used for building Bayesian time series models.
*   **(Optional) Kats:**  Another time series library from Facebook, offering a wider range of tools than Prophet, including forecasting, anomaly detection, and feature extraction.
*  **(Optional) tsfresh:** A library for automated feature extraction from time series data.

**Datasets Considered:**

The datasets used in this session will cover a range of characteristics to test the libraries' capabilities. Examples include:

*   **Air Passengers:** A classic dataset with a clear trend and seasonality, ideal for testing forecasting models.  (Often included in library examples).
*   **Energy Consumption Data:**  Data on electricity or gas consumption, often exhibiting daily and weekly seasonality.
*   **Stock Prices:**  Financial time series data, which can be volatile and challenging to predict.
*   **Weather Data:** Temperature, precipitation, or other weather-related data, showcasing seasonality and potential anomalies.
*   **Synthetic Data:**  We'll generate synthetic time series data with specific characteristics (e.g., different noise levels, trends, seasonality) to test the robustness of the models.
*   **UCI Machine Learning Repository Time Series Datasets:** Exploring datasets from this repository to find diverse and well-documented examples. ([https://archive.ics.uci.edu/](https://archive.ics.uci.edu/))
* **Kaggle datasets**: Exploring datasets in Kaggle ([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets?tags=13208-Time+Series))

**Code Organization:**

The code will be organized into Jupyter notebooks, with separate notebooks for:

*   `darts_tutorial.ipynb`:  A comprehensive tutorial on using Darts.
*   `library_comparison.ipynb`:  Comparing Darts with other libraries.
*   `dataset_exploration.ipynb`:  Exploring and preprocessing different datasets.
*   `models/<model_name>.ipynb`: If specific, complex models are built, they might have their own notebooks.
*   `data/`:  This directory will store the datasets used in the session (or scripts to download them).

**Requirements:**

*   Python 3.7+
*   Jupyter Notebook or JupyterLab
*   Install the necessary libraries:  `pip install darts statsmodels prophet scikit-learn ...` (A `requirements.txt` file will be provided).

**Further Exploration:**

*   **Advanced Darts Features:**  Explore ensemble models, hierarchical forecasting, and custom models in Darts.
*   **Deep Learning Models:**  Experiment with PyTorch Forecasting or TensorFlow Probability for building neural network-based time series models.
*   **Anomaly Detection:**  Investigate anomaly detection techniques using the libraries discussed.
*   **Feature Engineering:**  Utilize tsfresh or other feature engineering methods to improve model performance.
*   **Real-world Applications:**  Apply the learned techniques to real-world time series problems in your domain of interest.

This README provides a roadmap for the session and a foundation for further exploration in the field of time series analysis.  The code and notebooks in this repository will serve as a practical guide and reference for working with various time series libraries and datasets.