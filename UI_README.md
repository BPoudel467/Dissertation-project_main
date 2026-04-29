# IDS Comparative Study - Streamlit UI

A web-based user interface for the Intrusion Detection System comparative study.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

### 2. Run the UI
```bash
streamlit run app.py
```

### 3. Open in Browser
Navigate to `http://localhost:8501` in your web browser.

## Features

| Page | Description |
|------|-------------|
| **Home** | Overview of the project and quick actions |
| **Run Pipeline** | Execute the complete ML pipeline |
| **Results** | View supervised and unsupervised model results |
| **Model Comparison** | Compare performance across all models |
| **Visualizations** | View generated plots and charts |
| **About** | Project information and documentation |

## Project Structure

```
ids-comparative-study/
├── app.py                 # Streamlit UI application
├── main.py               # Original CLI pipeline
├── requirements.txt      # Core dependencies
├── requirements-ui.txt   # UI dependencies
└── outputs/
    ├── figures/          # Generated visualizations
    ├── models/           # Trained model files
    └── tables/           # Result CSV files
```

## UI Pages

### Home
- Project overview
- Quick action buttons
- Key metrics display

### Run Pipeline
- Step-by-step pipeline information
- Execute button with progress tracking

### Results
- Supervised model results (RF, SVM, XGBoost)
- Unsupervised model results (K-Means, Autoencoder)

### Model Comparison
- Side-by-side model performance
- Comparison visualizations

### Visualizations
- Confusion matrices
- ROC curves
- Feature importance plots
- Training metrics

## Notes

- The pipeline may take several minutes to complete
- Ensure the dataset is available in `data/raw/`
- Results will be saved to `outputs/` directory