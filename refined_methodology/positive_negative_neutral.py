"""
sentiment_analysis.py

Apply Hugging Face sentiment analysis to your yearly Nexis CSVs
(e.g. nexis_2021_structured.csv → nexis_2021_sentiment.csv).

Assumes each CSV has columns: year, title, body.

Dependencies:
    pip install pandas transformers torch
"""

import glob
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def clean_body(text: str) -> str:
    """
    Strip leading 'Dateline:' lines and the 'Body' header so
    only the article text remains.
    """
    if not isinstance(text, str):
        return ""
    # drop any 'Dateline:' line
    lines = [line for line in text.splitlines() if not line.startswith("Dateline:")]
    text = "\n".join(lines)
    # split off the 'Body' header if present
    if "Body" in text:
        _, _, text = text.partition("Body")
    return text.strip()

def truncate_text(text, tokenizer, max_length=450):
    """Truncate text to fit within model token limits"""
    # Tokenize with truncation
    encoded = tokenizer(text, truncation=True, max_length=max_length)
    # Decode back to text
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

def main():
    # 1) initialize the tokenizer and sentiment pipeline with 3-class model
    # Switch to a 3-class model that outputs POSITIVE / NEUTRAL / NEGATIVE
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,  # we only want the top label
        truncation=True,
        max_length=512
    )

    # 2) find all your yearly structured CSVs
    files = sorted(glob.glob("refined_methodology/ethical_*_structured.csv"))
    if not files:
        print("No files matching pattern found.")
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir("refined_methodology"))
        return

    # 3) process each file
    for infile in files:
        print(f"✔️  Processing {infile}…")
        df = pd.read_csv(infile)
        if "body" not in df.columns:
            print(f"❌  Skipping {infile}: missing 'body' column.")
            continue

        # 3a) clean up the body text
        df["clean_body"] = df["body"].apply(clean_body)

        # 3b) truncate text to fit model constraints
        print("Preparing text for sentiment analysis...")
        docs = df["clean_body"].fillna("").apply(
            lambda x: truncate_text(x, tokenizer, max_length=450)
        ).tolist()
        
        # Process in smaller batches to avoid memory issues
        batch_size = 8
        results = []
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        print(f"Processing {len(docs)} articles in {total_batches} batches...")
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            batch_results = sentiment_pipe(batch)
            results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{total_batches}")

        # 3c) append the results
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]

        # 3d) save to a new CSV
        base = os.path.splitext(infile)[0]
        outfile = f"{base}_sentiment.csv"
        df.to_csv(outfile, index=False)
        print(f"✅  Saved sentiment to {outfile}\n")
        
        # 3e) print some summary statistics
        sentiment_counts = df["sentiment_label"].value_counts(normalize=True) * 100
        print("\nSentiment Distribution:")
        for label, percentage in sentiment_counts.items():
            print(f"  {label}: {percentage:.1f}%")

if __name__ == "__main__":
    main()