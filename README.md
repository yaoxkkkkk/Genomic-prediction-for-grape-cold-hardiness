# Genomic Prediction for Grape Cold Hardiness

This project provides machine learning models for genomic prediction of grape cold hardiness.

## Usage

1. **Prepare your data**  
   Format your input data as `test_data.csv` (CSV format).

2. **Run the prediction script**  
   Use the following command to make predictions:

   ```bash
   python script.py \
       --model_dir /path/to/model/ \
       --test_data test_data.csv \
       --output_dir /output/path/ \
       --global_scaler global_scaler.pkl
    ```