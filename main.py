import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os

# Load the data
def load_data(file_path):
    print("Loading JSON data...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Rename columns
    df['wallet'] = df['userWallet']

    # Extract numeric 'amount' from actionData dictionary
    df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) / 1e6)  # Assuming USDC (6 decimals)

    print(f"Loaded {len(df)} transactions.")
    print("Columns:", df.columns.tolist())
    return df

# Feature engineering
def feature_engineering(df):
    print("Extracting wallet features...")
    grouped = df.groupby('wallet')
    features = []

    for wallet, group in grouped:
        deposit_count = len(group[group['action'] == 'deposit'])
        borrow_count = len(group[group['action'] == 'borrow'])
        repay_count = len(group[group['action'] == 'repay'])
        liquidation_count = len(group[group['action'] == 'liquidationcall'])
        deposit_amount = group[group['action'] == 'deposit']['amount'].sum()
        borrow_amount = group[group['action'] == 'borrow']['amount'].sum()
        
        borrow_deposit_ratio = borrow_count / deposit_count if deposit_count > 0 else 1.0

        features.append({
            'wallet': wallet,
            'deposit_count': deposit_count,
            'borrow_count': borrow_count,
            'repay_count': repay_count,
            'liquidation_count': liquidation_count,
            'deposit_amount': deposit_amount,
            'borrow_amount': borrow_amount,
            'borrow_deposit_ratio': borrow_deposit_ratio
        })

    features_df = pd.DataFrame(features)
    print(f"Generated features for {len(features_df)} wallets.")
    return features_df

# ML-based scoring
def ml_credit_scoring(features_df):
    print("Scoring wallets using Isolation Forest...")
    numeric_features = features_df.drop(columns=['wallet']).replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(numeric_features)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    anomaly_scores = model.decision_function(X_scaled)
    features_df['anomaly_score'] = anomaly_scores

    score_norm = MinMaxScaler(feature_range=(0, 1000))
    features_df['credit_score'] = score_norm.fit_transform(anomaly_scores.reshape(-1, 1))

    credit_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    credit_scores = (credit_scores * 1000).astype(int)

    features_df['credit_score'] = credit_scores

    print("Credit Score Summary Statistics:")
    print(features_df['credit_score'].describe())

    return features_df

# Save scores
def save_scores(df, output_file):
    df[['wallet', 'credit_score']].to_csv(output_file, index=False)
    print(f"Saved wallet scores to '{output_file}'.")

# Plot distribution
def plot_distribution(df):
    print("Plotting score distribution...")
    plt.figure(figsize=(10, 6))
    bins = range(0, 1100, 100)
    plt.hist(df['credit_score'], bins=bins, edgecolor='black')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.title('Wallet Credit Score Distribution')
    plt.xticks(bins)
    plt.grid(True)
    plt.savefig('score_distribution.png')
    print("Saved 'score_distribution.png'.")

# Main function
def main():
    input_file = 'user-wallet-transactions.json'  # Make sure this name matches your actual file
    output_file = 'wallet_scores.csv'

    if not os.path.exists(input_file):
        print(f"ERROR: File '{input_file}' not found.")
        return

    df = load_data(input_file)
    features_df = feature_engineering(df)
    scored_df = ml_credit_scoring(features_df)
    save_scores(scored_df, output_file)
    plot_distribution(scored_df)
    print("✅ Done.")

if __name__ == '__main__':
    main()

