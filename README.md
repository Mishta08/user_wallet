# Aave V2 Wallet Credit Scoring Model

## Overview
This project assigns a credit score between 0 and 1000 to each wallet interacting with the Aave V2 protocol using historical transaction data. A higher score indicates more responsible, lower-risk behavior, while a lower score flags potential risky, bot-like, or exploitative actions.

### 1. Load and Parse Data
Reads the transaction JSON file containing:
- userWallet
- action (deposit, borrow, repay, etc.)
- amount
- asset type
- timestamp

### 2. Feature Engineering
Creates features like:
- Total transactions per wallet
- Action ratios (e.g., borrow/deposit)
- Average transaction value
- Number of unique assets used

### 3. ML Model
Uses **Isolation Forest** to detect anomalies.
- Anomaly score → Scaled to 0–1000 credit score
- Higher score = safer user
- Lower score = suspicious behavior

### 4. Outputs:
 - wallet_scores.csv – final scores
 - score_distribution.png – visual chart of scores

### Pipeline Architecture
1. **Data Loading:** Read and parse the JSON file.
2. **Feature Extraction:** Compute features for each wallet.
3. **Score Assignment:** Apply the rule-based function to compute the credit score.
4. **Output Generation:** Save scores into a CSV file and generate a distribution plot.


## Setup and Execution

### Requirements
- Python 3.6+
- Required libraries: `pandas`, `numpy`, `matplotlib`

Install dependencies with:
```bash
pip install pandas numpy matplotlib
