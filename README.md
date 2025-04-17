## 📁 Project Structure

```text
agitrader/
├── main.ipynb              # Central Colab notebook for training, logging, evaluation
├── README.md               # Project overview, structure, usage instructions
├── requirements.txt        # pip dependencies (e.g., gym, sb3, pandas, etc.)
├── .gitignore              # Files to exclude from Git version control
│
├── data/                   # Data loading, parsing, and selection tools
│   ├── raw/                # (Optional) Unprocessed IEX DEEP/TOPS data files
│   │   └── iex_aapl_2021-03-15.csv
│   ├── parsed/             # Downsampled LOB data used in training
│   │   └── lob_day1_aapl.csv
│   ├── preprocess.py       # IEX DEEP parser and top-5 LOB downsampler
│   └── selector.py         # Selects eventful/volatile days for training
│
├── env/                    # Gym-compatible trading environment
│   ├── __init__.py
│   ├── lob_env.py          # Main environment class (reset, step, space definitions)
│   ├── state_builder.py    # Constructs LOB observation feature vectors
│   └── fills.py            # Simulates order fills, queueing, and latency
│
├── agent/                  # PPO + LSTM agent logic and training
│   ├── __init__.py
│   ├── policy.py           # Optional custom LSTM or Transformer policy
│   ├── train.py            # PPO training loop and agent initialization
│   └── callbacks.py        # Logging, checkpointing, and evaluation hooks
│
├── reward/                 # Reward signal components and shaping logic
│   ├── __init__.py
│   └── reward.py           # Implements spread capture, drawdown, inventory penalties
│
├── evaluation/             # Post-training strategy discovery and visualization
│   ├── __init__.py
│   ├── logger.py           # Logs episode data: actions, prices, rewards, etc.
│   ├── cluster.py          # Performs t-SNE/PCA and strategy clustering
│   └── plot.py             # Visualizations of PnL, actions, and behavioural clusters
│
└── config/                 # Global settings and experiment constants
    ├── __init__.py
    └── settings.py         # Defines sampling rate, asset symbol, sequence length, etc.
