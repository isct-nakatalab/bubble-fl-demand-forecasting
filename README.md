# Prediction Performance-based Grouping for Bubble-based Federated Learning in Retail Demand Forecasting

## init

```bash
git clone https://github.com/isct-nakatalab/bubble-fl-demand-forecasting

cd bubble-fl-demand-forecasting

uv sync
```

Place [M5 Dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) under `data/`

## Commands

### Data preprocess

```bash
uv run python -m data.preprocess_m5
```

### Experiments

```bash
# LM and CM
uv run python -m baselines.local_learning
uv run python -m baselines.centralized

# Baselines
uv run python -m baselines.fedavg.run

# Proposed
uv run python -m proposed.run
```
