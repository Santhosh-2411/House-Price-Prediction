# House Price Prediction – Boston Housing Dataset

Machine-learning case study: predict house prices (`MEDV`, median value in $1000s) from 13 housing /
neighbourhood features using **Linear Regression**, **Decision Tree Regression** and
**Random Forest Regression**, with a **Flask web application**.

## Project structure
```
house_price_prediction/
├── app.py               # Flask web app (predict page, performance page, JSON API)
├── house_price_models.ipynb  # Jupyter notebook: simple code + graphs for all 3 models
├── train.py             # trains + evaluates all models, saves models and charts
├── download_data.py     # downloads the real Boston dataset to data/boston.csv
├── requirements.txt
├── src/
│   ├── config.py        # paths, feature names, descriptions
│   ├── data_loader.py   # loads dataset (CSV -> OpenML -> synthetic fallback)
│   └── train_models.py  # EDA, preprocessing, tuning, evaluation, plots
├── templates/           # index.html, performance.html, base.html
├── static/              # css + generated plots
├── data/                # boston.csv goes here
└── models/              # saved .joblib models + metrics.json (created by training)
```

## Run in VS Code
1. Unzip and open the folder in VS Code (**File → Open Folder**).
2. Open a terminal (**Ctrl + `**) and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
3. Install packages: `pip install -r requirements.txt`
4. *(Recommended, needs internet)* get the real dataset: `python download_data.py`
   - or place your own `boston.csv` (columns CRIM … LSTAT, MEDV) inside `data/`.
5. Train the models: `python train.py`
6. Start the web app: `python app.py` → open **http://127.0.0.1:5000**

If you skip steps 4–5, `python app.py` trains automatically on first start.

## Jupyter notebook (graphs for each model)
1. In VS Code install the **Python** and **Jupyter** extensions (Extensions tab, Ctrl+Shift+X).
2. Open `house_price_models.ipynb`.
3. Click **Select Kernel** (top right) and choose the `venv` Python you created.
4. Click **Run All**.

It shows, for Linear Regression, Decision Tree and Random Forest: a simple RM-vs-price graph with the
model line, the full 13-feature model with metrics, actual-vs-predicted and feature graphs, the
decision tree drawing, trees from inside the random forest, and a final comparison of all three.

## What is implemented
- Data loading, missing-value handling (median imputation), feature scaling (linear model)
- EDA charts: price distribution, correlation heatmap, key-feature scatter plots
- 80/20 train/test split, GridSearchCV tuning (Decision Tree, Random Forest), 5-fold cross-validation
- Metrics: MAE, MSE, RMSE, R² (train / test / CV); best model chosen by test R²
- Charts: actual vs predicted, model comparison, feature importance, residuals
- Web app: form with all 13 features, model selection, prediction in dollars, side-by-side
  predictions from all three models, performance page, JSON endpoint `POST /api/predict`

## Notes
- Prices are 1970s Boston values, so treat predictions as a learning exercise only.
- scikit-learn removed `load_boston` (v1.2) because of ethical concerns with the `B` feature,
  which is why the data is loaded from CSV / OpenML instead.
- If no real dataset is available (e.g. offline), a clearly-labelled synthetic Boston-like
  dataset is used so the app still runs; the Performance page shows a warning when this happens.
