# House Sales Price Prediction

A portfolio project based on the **House Sales practical exam** completed as part of the **DataCamp Data Scientist Associate** certification.

The project demonstrates data-quality checks, missing-value handling, exploratory aggregation, and regression workflows using Python, pandas, and scikit-learn.

## Business problem

RealAgents wants to estimate a house's sale price from its characteristics so it can make better listing-price decisions and reduce time on the market.

## Project tasks

1. **Missing-value audit** — identify missing `city` values in `house_sales.csv`.
2. **Data cleaning and validation** — standardize house types and area values; handle missing fields according to the task specification; validate the required eight-column dataset.
3. **Exploratory analysis** — calculate average sale price and sample variance by number of bedrooms.
4. **Baseline model** — fit a Linear Regression model on `train.csv` and predict prices for `validation.csv`.
5. **Comparison model** — fit a Random Forest Regressor and compare predictions with the linear baseline.

## Tools

- Python
- pandas and NumPy
- scikit-learn
- Missing-value imputation and one-hot encoding
- Linear Regression and Random Forest Regression

## Repository structure

```text
.
├── data/
│   └── raw/                 # Add authorized course CSV files here locally
├── outputs/                 # Generated outputs (not committed)
├── src/
│   └── house_sales_project.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Data files

The DataCamp course data are **not included** in this repository. Do not upload course-provided data unless the license or course terms explicitly allow redistribution.

To run the project locally, place the authorized files in `data/raw/`:

- `house_sales.csv`
- `train.csv`
- `validation.csv`

The script expects the missing-value marker `--` used in the course dataset.

## Run

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python src/house_sales_project.py
```

The script writes the cleaned data, bedroom-price summary, and model predictions to `outputs/`. Generated CSVs and raw data are ignored by Git to avoid accidentally publishing course data.

## Validation notes

The cleaning step checks the expected columns, allowed city and house-type values, and remaining missing values. The modeling workflow fits preprocessing on the training data and applies the fitted transformations to validation data to avoid fitting encoders or imputers on validation rows.

**Scope:** This is a practical learning/portfolio project, not a deployed pricing service. Model quality should be assessed with an appropriate held-out target and metrics before using predictions for business decisions.

## Certification

DataCamp — **Data Scientist Associate**  
Obtained: November 3, 2026 · Expires: November 2, 2028

Credential: https://www.datacamp.com/certificate/DSA0018741766310

## Acknowledgment

The business scenario and task structure are based on the DataCamp House Sales practical exam. The repository contains a documented implementation, not the proprietary course datasets or a claim of official endorsement by DataCamp.
