UK-stock-DL/
├── data/
│   ├── raw/                 # Raw downloaded data
│   ├── processed/           # Cleaned and feature-engineered data
│   └── external/            # External datasets
├── notebooks/               # Jupyter notebooks for EDA and experiments
├── src/
│   ├── __init__.py
│   ├── data/                # Data loading & preprocessing scripts
│   │   ├── __init__.py
│   │   ├── fetch_data.py    # Download data from yfinance
│   │   ├── preprocess.py    # Clean and process data, calculate indicators
│   ├── features/            # Feature engineering
│   │   ├── __init__.py
│   │   └── technical_indicators.py
│   ├── models/              # Model definitions & training scripts
│   │   ├── __init__.py
│   │   ├── base_model.py    # Base model class
│   │   ├── model_cl.py      # CNN-LSTM model
│   │   ├── model_gru.py     # GRU model
│   │   └── train.py         # Training pipeline
│   ├── evaluation/          # Evaluation & metrics
│   │   ├── __init__.py
│   │   └── evaluate.py
│   └── utils/               # Utility scripts
│       ├── __init__.py
│       └── helpers.py
├── experiments/             # For keeping experiment logs, configs, results
├── configs/                 # YAML/JSON configs for models, training, etc.
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and instructions
├── .gitignore
└── setup.py                 # For pip-installable package