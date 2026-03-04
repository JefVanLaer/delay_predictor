# Shipping Delay Predictor

POC for predicting cargo vessel delays using AIS tracking data. The pipeline detects port visits, reconstructs voyages, engineers real-time features, and serves predictions from a trained gradient boosting model via a Kafka consumer.

## Data sources

| Source | Description |
|--------|-------------|
| [US Marine Cadastre (MCS)](https://hub.marinecadastre.gov/pages/vesseltraffic) | Vessel AIS position records — `data/ais/ais-YYYY-MM-DD.parquet` |
| [World Port Index (WPI)](https://msi.nga.mil/Publications/WPI) | 657 US port locations — `data/ports/ports.csv` |

AIS data is not included in the repo; download it from the MCS link above and convert to Parquet using `data/ais/ais_fetcher.py`.

## Project layout

```
src/
  methods.py                               # DMS → decimal-degree coordinate conversion
  port_matcher.py                          # Spatial port matching via GeoPandas + EPSG:3857 buffering
  voyage_creator.py                        # Port-visit detection and voyage labelling
  ais_stream_mock.py                       # Replay Parquet AIS data as a timed JSON stream
  ais_kafka_predictor.py                   # Kafka consumer that runs the full prediction pipeline

notebooks/
  exploration.ipynb                        # Initial EDA
  port_matching.ipynb                      # PortMatcher end-to-end demo
  voyage_creator.ipynb                     # Voyage reconstruction
  training.ipynb                           # Model training and evaluation
  ais_stream_mock_predictor_test.ipynb     # Integration test (mock stream → predictor)

models/
  delay_predictor.joblib                   # Trained scikit-learn pipeline
```

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/port_matching.ipynb
```

To run the Kafka predictor against live data:

```python
from src.ais_kafka_predictor import AISKafkaPredictor
predictor = AISKafkaPredictor("localhost:9092", topic="ais-stream", ports_gdf=gdf_ports)
predictor.run()
```

To replay recorded data without a broker, see `notebooks/ais_stream_mock_predictor_test.ipynb`.

## Model

Gradient boosting regressor predicting remaining voyage hours. 
Features: 
- Rolling SOG windows (6 h / 24 h)
- SOG deviation from lane average
- Remaining and total great-circle distance
- Voyage fraction completed. 

Evaluated with MAE and R².

## Next steps

- Add weather data to improve delay attribution
- Hyperparameter tuning and algorithm comparison
- Deploy as a real-time REST/gRPC service