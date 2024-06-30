#!/bin/bash

python3 /app/src/train_pipeline.py


python3 /app/src/predict.py


tail -f /dev/null