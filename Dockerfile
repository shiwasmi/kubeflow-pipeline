FROM python:3.10-slim

# Install core ML and data packages
RUN pip install --no-cache-dir \
    scikit-learn==1.5.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    requests==2.31.0 \
    joblib==1.4.2 \
    kfp==2.16.0
