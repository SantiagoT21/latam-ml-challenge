# PART 1: Bugs fixed on DS jupyter notebook `exploration.ipynb`

---

## 1. Virtual Environment Setup

To keep dependencies organized and avoid conflicts with other projects, I created a **Python 3.9 virtual environment**. This dedicated environment isolates all necessary packages and versions for this project:

```bash
python3.9 -m venv venv
source venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
```

I also made sure to include the xgboost library and any other missing dependencies (ipykernel, numpy, pandas, etc.) in the requirements-dev.txt file.

---

## 2. Bug Fixes and Improvements

### 2.1 `is_high_season` Function Bug

- Issue: The function did not consider time boundaries properly. For instance, is_high_season("2017-12-31 14:55:00") returned 0 even though it should have returned 1. The date ranges ended at "YYYY-MM-DD 00:00:00", excluding flights later on the same end date.
- Fix: I adjusted the function to ensure that flights on the upper-bound date are correctly classified if they fall within the “high season.”

---

### 2.2 Updating Seaborn’s `set`

- Issue: The code used `sns.set()`, which is now deprecated.
- Fix: I replaced it with `sns.set_theme()`, aligning with the current recommended approach in Seaborn.

---

### 2.3. Missing x and y in Seaborn Bar Plots

- Issue: All calls to `sns.barplot` were using positional arguments incorrectly, causing errors and confusion.
- Fix: I replaced the positional arguments with the explicit x= and y= arguments, aligning with the latest Seaborn requirements.

---

### 2.4. Removing Unused Variables

- Issue: A variable named `training_data` was defined but never used.
- Fix: I removed the redundant cell to avoid confusion and streamline the notebook.

---

### 2.5. Minor Style Changes

Issue: Inconsistent in the code format made the code less readable.
Fix: I applied minor formatting changes for clarity and consistency.

---

# PART 1: Model Analysis and Selection

As part of my role as an ML Engineer, I conducted a thorough analysis of several candidate models to predict flight delays (Trained by DS). In this challenge, the primary concern is the minority class—flights that experience **delays**—since the goal is to accurately identify when a flight is likely to be delayed. Below, I detail my approach, the metrics I considered, and the conclusions I reached regarding the best model to move forward with.

---

## 1. Overview

I worked with two model families—**XGBoost** and **Logistic Regression**—under multiple configurations that decide the DS in the notebook:

1. **Without Class Balancing**
2. **With Feature Importance**
3. **With/Without Class Balancing in combination with Feature Importance**

During the initial exploration, it became apparent that the dataset suffered from a notable class imbalance: the vast majority of flights were **on time** (class `0`), while delayed flights (class `1`) constituted a smaller fraction of the data. This imbalance caused the baseline models (those without any balancing strategy) to perform poorly on the delayed class.

---

## 2. Metrics Considered

I focused primarily on the **precision**, **recall**, and **F1-score** for the **delayed class** (class `1`), because:

- **Precision** indicates how often the model is correct when predicting a delay.
- **Recall** measures how many of the actual delayed flights the model successfully identifies.
- **F1-score** combines both precision and recall, serving as a balanced metric for imbalanced classification problems.

In real-world scenarios, missing a delay (low recall) can be costly, while frequently over-predicting delays (low precision) can also be problematic. Hence, an **F1-score for the delayed class** represents a balanced view of how effectively the model captures true delays while minimizing false alarms.

---

## 3. Model-by-Model Insights

### 3.1. Models Without Class Balancing (All Features)

- **Plain XGBoost**

  - The F1-score for the delayed class was almost **0.00**.
  - The model heavily favored the majority class (on-time flights), rarely predicting delays.

- **Plain Logistic Regression**
  - The F1-score for the delayed class stood around **0.06**.
  - Although slightly better than plain XGBoost, it still consistently misclassified the minority class.

#### Conclusion on Non-Balanced Models (All Features)

Without balancing, **both** XGBoost and Logistic Regression struggled to identify delayed flights. Their extremely low recall for class `1` indicated that most delayed flights were being overlooked.

---

### 3.2. Models With Feature Importance but **Without** Class Balancing

- **XGBoost (Feature Importance, No Balance)**

  - F1-score for the delayed class was about **0.01**, which is still extremely poor.
  - Despite using only the top predictive features, the model continued to favor class `0` heavily.

- **Logistic Regression (Feature Importance, No Balance)**
  - F1-score for the delayed class was about **0.03**, slightly higher than XGBoost but still very low.
  - Similar to the previous scenario, the imbalance remained a major obstacle; the model found it safer to classify flights as “on-time.”

#### Conclusion on Feature-Importance-Only Models

Even after focusing on a refined set of top features, the **lack of class balancing** severely impacted the models’ ability to detect the delayed class. The F1-scores (0.01 for XGBoost and 0.03 for Logistic Regression) reflect how the models still predicted “no delay” the vast majority of the time.

---

### 3.3. Models With Feature Importance **and** Class Balancing

- **XGBoost (With Balance + Feature Importance)**

  - Confusion matrix indicated significantly more delayed flights being captured.
  - F1 for the delayed class improved markedly to about **0.36**.

- **Logistic Regression (With Balance + Feature Importance)**
  - Achieved an F1 for the delayed class also around **0.36**.
  - Demonstrated virtually the same performance as the balanced XGBoost model in terms of F1.

#### Conclusion on Balanced + Feature Importance Models

By balancing the classes, both XGBoost and Logistic Regression saw a **substantial jump** in the F1-score for the delayed class, going from near zero to approximately **0.36**. Although the overall accuracy dropped slightly, it is a typical and acceptable trade-off for detecting more of the minority class.

---

## 4. Final Model Decision

Given that both balanced **XGBoost** and **Logistic Regression** displayed **similar F1-scores** (about 0.36 for class `1`), I factored in additional considerations:

1. **Interpretability**

   - Logistic Regression is more transparent, allowing me (and stakeholders) to easily understand how coefficients influence the probability of a delay.
   - XGBoost, being an ensemble method, is more complex and less straightforward to interpret, although feature importance helps somewhat.

2. **Speed and Complexity**

   - Logistic Regression typically requires fewer computational resources and provides faster inference times.
   - XGBoost can handle nonlinear relationships and large datasets efficiently, but it generally has longer training and inference times compared to a simple linear model.

3. **Production Readiness**
   - Due to its simplicity, Logistic Regression is often easier to deploy and maintain.
   - If I required higher performance in the future or had much larger datasets, XGBoost could be a better fit. For now, however, both models deliver similar F1-scores.

#### My Recommendation

Although both balanced models perform similarly in terms of F1 for the delayed class, **Logistic Regression with class balancing** is my preferred choice for immediate production deployment because:

- It is **easier to interpret**, vital when explaining decisions to non-technical stakeholders (e.g., airline staff and operations teams).
- It tends to be **faster** in both training and prediction, which is beneficial if this service needs to scale or handle frequent real-time queries.

---

## 5. Conclusion

By focusing on **F1-score** for the delayed class, I identified that **Logistic Regression with class balancing** (and feature selection) performs as well as XGBoost in identifying delayed flights, while offering clearer interpretability and lower computational costs. Consequently, I chose Logistic Regression for its balance of performance, speed, and simplicity—key advantages in a real-world production environment dealing with flight operations.


# PART 2: Bug Fixes in `test_model.py`

Previously, our test file `test_model.py` attempted to load the CSV file using the path `"../data/data.csv"`. However, when running tests from the repository's root directory, the relative path did not match the actual location of the data file. 

To resolve this, we changed the path to `"./data/data.csv"`. This way, the file is correctly recognized regardless of the current working directory (assuming we run tests from the root). 

Now, we can successfully execute the tests with:
```bash
python -m unittest .\tests\model\test_model.py
# and
make model-test
```

# PART 2: Updates to requirements.txt
While running the API tests, the following error was encountered:
```bash
AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
```
To address this, we updated the dependencies in requirements.txt to the following versions:

- `fastapi~=0.111.0`
- `pydantic~=1.10.17`
- `uvicorn~=0.30.1`

These updates ensure compatibility with the testing framework and resolve the above error.

Now, we can successfully execute the tests with:
```bash
python -m unittest .\tests\api\test_api.py
#and
make api-test
```

# PART 3: How to Deploy a Docker Container to Cloud Run

## 1. Build the Docker image
```bash
docker build -t gcr.io/[PROJECT-ID]/[IMAGE-NAME]:[TAG] .
```

---

## 2. Authenticate Docker with Google Cloud
```bash
gcloud auth configure-docker
```

---

## 3. Push the image to Google Container Registry
```bash
docker push gcr.io/[PROJECT-ID]/[IMAGE-NAME]:[TAG]
```

---

## 4. Deploy to Cloud Run
```bash
gcloud run deploy [SERVICE-NAME] \
    --image gcr.io/[PROJECT-ID]/[IMAGE-NAME]:[TAG] \
    --region [REGION] \
    --platform managed \
    --allow-unauthenticated
```

# PART 4: Setting Up Secrets for Continuous Delivery Workflow in Google Cloud

---

## Prerequisites

Before starting, ensure you have the following:

- Access to a Google Cloud account.
- Administrative permissions to create a service account in GCP.
- A GitHub repository where the workflow will be implemented.

---

## Step 1: Create a Service Account

1. **Access the GCP Console**:
   - Open the [Google Cloud Console](https://console.cloud.google.com).

2. **Navigate to IAM & Admin > Service Accounts**:
   - In the left-hand menu, go to **IAM & Admin** and select **Service Accounts**.

3. **Create a New Service Account**:
   - Click on **Create Service Account**.
   - Provide a descriptive name, e.g., `github-actions-service-account`.

4. **Assign Permissions**:
   - Grant the following roles to the service account:
     - **Storage Object Viewer**: To download files from Google Cloud Storage.
     - **Cloud Build Editor**: To build Docker images.
     - **Cloud Run Admin**: To deploy services to Cloud Run.
     - **Artifact Registry Reader**: To access Docker images stored in Artifact Registry.

5. **Finalize and Save**:
   - Complete the process and create the service account.

---

## Step 2: Generate Service Account Credentials

1. **Go to Service Account Management**:
   - Navigate to the created service account (`github-actions-service-account`).

2. **Generate a Key**:
   - Click on the three dots next to the service account and select **Manage Keys**.
   - Click on **Add Key > Create New Key**.
   - Select the **JSON** format and download the key file.

3. **Store the JSON Securely**:
   - Save this file as it contains the credentials required for GitHub Actions.

---

## Step 3: Add Secrets to GitHub Repository

1. **Access GitHub Repository Settings**:
   - Go to your repository in GitHub.
   - Navigate to **Settings > Secrets and variables > Actions**.

2. **Add Secrets**:
   - Click on **New repository secret** for each secret required by the workflow.

3. **Secrets to Add**:
   - **`GCP_CREDENTIALS`**: Paste the contents of the JSON key file downloaded earlier.
   - **`MODEL_VERSION`**: Specify the version of the model to download (e.g., `v1.0.0`).
   - **`GCP_REGION`**: Specify the region for GCP services (e.g., `us-central1`).
   - **`GCP_PROJECT_ID`**: Provide the Project ID of your GCP project.
   - **`GCP_IMAGE_NAME`**: Specify the Docker image name (e.g., `delay-service`).

---

## Step 4: Validate the Workflow

1. **Verify Secrets in GitHub**:
   - Ensure all secrets are correctly added to your GitHub repository.

2. **Run the Workflow**:
   - Trigger the workflow by pushing changes to the specified branches (`main` or `release/*`).

3. **Check Logs**:
   - Monitor the workflow logs in GitHub Actions to ensure the deployment is successful.

---

## Example of Secrets Configuration

| Secret Name         | Description                            | Example Value            |
|---------------------|----------------------------------------|--------------------------|
| `GCP_CREDENTIALS`   | JSON credentials for service account   | JSON file contents       |
| `MODEL_VERSION`     | Version of the model to deploy         | `v1.0.0`                |
| `GCP_REGION`        | GCP region for deployment              | `us-central1`           |
| `GCP_PROJECT_ID`    | ID of the GCP project                 | `my-gcp-project-id`     |
| `GCP_IMAGE_NAME`    | Name of the Docker image              | `delay-service`         |