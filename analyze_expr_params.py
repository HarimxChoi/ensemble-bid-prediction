import pandas as pd

def analyze_params():
    try:
        df = pd.read_csv("mccp_simulation/dataset/raw_dataset/CS_BIDS_EXPR_202601061507.csv", encoding='cp949')
        
        # Select parameter columns
        cols = ['JUDGE_BASE', 'JUDGE_SEQ', 'BID_VALUE1', 'BID_VALUE2', 'BID_VALUE3', 'BASE_VALUE1']
        df = df[cols]
        
        # Drop duplicates to see unique formulas
        unique_formulas = df.drop_duplicates().sort_values(['JUDGE_BASE', 'JUDGE_SEQ'])
        
        print(f"Total Unique Parameter Sets: {len(unique_formulas)}")
        print("-" * 80)
        print(f"{'BASE':<5} {'SEQ':<5} {'V1':<10} {'V2':<5} {'V3':<5} {'BASE_VAL':<10}")
        print("-" * 80)
        
        for _, row in unique_formulas.iterrows():
            print(f"{row['JUDGE_BASE']:<5} {row['JUDGE_SEQ']:<5} {row['BID_VALUE1']:<10} {row['BID_VALUE2']:<5} {row['BID_VALUE3']:<5} {row['BASE_VALUE1']:<10}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_params()
