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
│   │   ├── README.md
│   │   ├── fetch_data.py    # Download data from yfinance & fetch news
│   │   ├── preprocess.py    # Clean and process data, technical indicators & sentiment merge
│   ├── features/            # Feature engineering
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── technical_indicators.py
│   │   └── sentiment.py     # News sentiment analysis using VADER
│   ├── models/              # Model definitions & training scripts
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── base_model.py    # Base model class
│   │   ├── model_cl.py      # CNN-LSTM model
│   │   ├── model_gan.py     # GAN model (Generator & Discriminator)
│   │   ├── model_gru.py     # GRU model
│   │   ├── model_var.py     # VAR model
│   │   └── train.py         # Data splitting utility
│   ├── evaluation/          # Evaluation & metrics
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── evaluate.py
│   └── utils/               # Utility scripts
│       ├── __init__.py
│       ├── README.md
│       ├── helpers.py
│       └── plotting.py
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and instructions
├── .gitignore
└── LICENSE
