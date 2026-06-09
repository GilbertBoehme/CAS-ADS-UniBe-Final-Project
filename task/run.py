import sys
import os

BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, BASE_DIR)

from module.ingest import ingest
from module.eda import run_eda
from module.model import train_pipeline, test_pipeline

# COMMAND ----------

# DBTITLE 1,Step 1: Ingest data
# Point to the data directory containing *_openmeteo.csv files
DATA_DIR = os.path.join(BASE_DIR, "data")

df = ingest(DATA_DIR)
df.head()

# COMMAND ----------

# DBTITLE 1,Step 2: Exploratory Data Analysis
df_clean = run_eda(df)

# COMMAND ----------

# DBTITLE 1,Step 3: Model training and evaluation
# Train both models using tuned hyperparameters (hidden=64, layers=1, dropout=0.40, lr=0.0005)
# Outputs: loss curves, validation threshold metrics (no val plots)
artifacts = train_pipeline(df)

# COMMAND ----------

# Evaluate both models on held-out test set (2023+)
# Outputs: test confusion matrices, ROC curves, PR curves, threshold analysis, comparison table
results = test_pipeline(artifacts)