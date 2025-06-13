import polars as pl
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from pprint import pprint
import time
import humanize


def log(message):
    """Logs a message with the current timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

log("Script started")

# Generate random data for 50,00,000 rows
num_rows = 50_00_000
start_date = datetime(2016, 7, 18)

log("Initializing data generation")
start_time = time.time()

# Using tqdm to track progress
data = {
    "Cust Id": np.random.choice([f"C101{str(i).zfill(3)}" for i in range(1, 100)], size=num_rows),
    "Gender": np.random.choice(["Male", "Female"], size=num_rows),
    "Age": np.random.randint(18, 70, size=num_rows),
    "Segment": np.random.choice(["HUF", "Individual", "Corporate"], size=num_rows),
    "Pincode": np.random.randint(100000, 999999, size=num_rows),
    "Region": np.random.choice(["North", "South", "East", "West"], size=num_rows),
    "Loan Account Number": np.random.choice([f"HL{str(i).zfill(5)}" for i in range(1, 1000)], size=num_rows),
    "Loan Product Id": np.random.choice(["HL", "PL", "AL"], size=num_rows),
    "Loan Amount": np.random.randint(1_00_000, 1_00_00_000, size=num_rows),
    "Loan Tenure": np.random.choice([120, 180, 240, 300], size=num_rows),
    "Loan Start Date": [start_date + timedelta(days=np.random.randint(0, 3650)) for _ in range(num_rows)],
    "IRR": np.round(np.random.uniform(7, 15, size=num_rows), 2),
    "EMI Amount": np.random.randint(5_000, 1_00_000, size=num_rows),
    "Repayment Day": np.random.choice([5, 10, 15, 20], size=num_rows),
    "EMI Start Date": [start_date + timedelta(days=np.random.randint(30, 3650)) for _ in range(num_rows)],
    "Repayment Date": [start_date + timedelta(days=np.random.randint(30, 3650)) for _ in range(num_rows)],
    "Repayment Month": np.random.randint(1, 12, size=num_rows),
    "Default": np.random.choice(["Normal Payment", "Excess Payment", "Payment Default"], size=num_rows),
}



log("Creating Polars DataFrame")
df = pl.DataFrame(data)

log("Writing DataFrame to CSV")

end_time = time.time()
df.write_csv(f"large_data {humanize.intword(num_rows).replace('', '')}.csv")
log(f"Script completed in {end_time - start_time:.2f} seconds")
