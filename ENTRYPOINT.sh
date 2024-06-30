#!/bin/bash

# Run the training script
python3 /app/src/train_pipeline.py

# Run the prediction script
python3 /app/src/predict.py

# Keep the container running
tail -f /dev/null