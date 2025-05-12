"""
ai_ethics_analysis.py

Comprehensive analysis of AI ethics articles from Nexis, including:
- Sentiment/tone (positive, neutral, negative)
- Ethical issues (privacy, bias, etc.)
- Ethical principles (utilitarianism, etc.)
- Recommendations
- AI technologies mentioned

Outputs an enhanced CSV with all analysis results.
"""

import glob
import os
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Define keyword lists for each category
ETHICAL_ISSUES = [
    "privacy", "surveillance", "bias", "discrimination", "fairness", 
    "transparency", "explainability", "accountability", "job loss", "unemployment",
    "safety", "security", "human rights", "autonomy", "agency", "consent",
    "manipulation", "misinformation", "disinformation", "digital divide",
    "inclusivity", "access", "inequality", "environmental impact", "power concentration"
]

ETHICAL_PRINCIPLES = [
    "utilitarianism", "utilitarian", "greatest good", "greatest happiness",
    "deontology", "deontological", "categorical imperative", "kant", "kantian",
    "virtue ethics", "aristotle", "aristotelian", "character", "virtues",
    "justice", "fairness", "equity", "equality", "rights based", "autonomy",
    "dignity", "beneficence", "non-maleficence", "justice", "social contract",
    "consequentialism", "consequentialist"
]

RECOMMENDATIONS = [
    "regulation", "policy", "governance", "oversight", "standards", "guidelines",
    "framework", "legislation", "law", "compliance", "audit", "certification",
    "should", "must", "need to", "recommend", "propose", "suggest", "advocate",
    "calls for", "urges", "emphasizes the importance", "highlight the need"
]

AI_TECHNOLOGIES = [
    "machine learning", "deep learning", "neural network", "generative ai",
    "large language model", "llm", "chatgpt", "gpt", "bert", "transformer", 
    "computer vision", "facial recognition", "image recognition",
    "natural language processing", "nlp", "text generation", "speech recognition",
    "autonomous vehicle", "self-driving car", "drone", "robot", "robotics",
    "recommendation system", "algorithm", "predictive analytics", "data mining",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    "ai model", "decision-making system"
]

def clean_body(text: str) -> str:
    """Clean up the article text"""
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
    encoded = tokenizer(text, truncation=True, max_length=max_length)
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

def find_matches(text, keyword_list):
    """Find all matches of keywords in the text"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    text = text.lower()
    matches = []
    
    for keyword in keyword_list:
        # Use word boundary to find whole words/phrases
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text):
            matches.append(keyword)
    
    return matches

def map_sentiment_to_tone(sentiment_label):
    """Map sentiment labels to tone categories"""
    if sentiment_label == "POSITIVE":
        return "enthusiastic"
    elif sentiment_label == "NEGATIVE":
        return "critical"
    else:  # NEUTRAL
        return "balanced/neutral"

def main():
    # 1) initialize the sentiment model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        truncation=True,
        max_length=512
    )

    # 2) find all structured CSVs
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

        # Clean up the body text
        df["clean_body"] = df["body"].apply(clean_body)

        # Analyze sentiment (tone)
        print("Preparing text for sentiment analysis...")
        docs = df["clean_body"].fillna("").apply(
            lambda x: truncate_text(x, tokenizer, max_length=450)
        ).tolist()
        
        # Process in batches
        batch_size = 8
        results = []
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        print(f"Processing {len(docs)} articles in {total_batches} batches...")
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            batch_results = sentiment_pipe(batch)
            results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}/{total_batches}")

        # Add sentiment results
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
        
        # Map sentiment to tone categories
        df["tone"] = df["sentiment_label"].apply(map_sentiment_to_tone)
        
        # Analyze for ethical issues, principles, recommendations, and technologies
        print("Analyzing ethical content...")
        df["ethical_issues"] = df["clean_body"].apply(lambda x: find_matches(x, ETHICAL_ISSUES))
        df["ethical_principles"] = df["clean_body"].apply(lambda x: find_matches(x, ETHICAL_PRINCIPLES))
        df["recommendations"] = df["clean_body"].apply(lambda x: find_matches(x, RECOMMENDATIONS))
        df["technologies"] = df["clean_body"].apply(lambda x: find_matches(x, AI_TECHNOLOGIES))
        
        # Add counts for easier analysis
        df["issue_count"] = df["ethical_issues"].apply(len)
        df["principle_count"] = df["ethical_principles"].apply(len)
        df["recommendation_count"] = df["recommendations"].apply(len)
        df["technology_count"] = df["technologies"].apply(len)

        # Save to a new CSV
        base = os.path.splitext(infile)[0]
        outfile = f"{base}_analysis.csv"
        df.to_csv(outfile, index=False)
        print(f"✅  Saved comprehensive analysis to {outfile}\n")
        
        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Total articles: {len(df)}")
        
        # Tone distribution
        print("\nTone Distribution:")
        tone_counts = df["sentiment_label"].value_counts(normalize=True) * 100
        for tone, percentage in tone_counts.items():
            print(f"  {tone}: {percentage:.1f}%")
        
        # Most common issues
        all_issues = []
        for issues in df["ethical_issues"]:
            all_issues.extend(issues)
        
        if all_issues:
            from collections import Counter
            top_issues = Counter(all_issues).most_common(5)
            print("\nTop Ethical Issues Mentioned:")
            for issue, count in top_issues:
                print(f"  {issue}: {count} articles ({count/len(df)*100:.1f}%)")
        
        # Most common technologies
        all_techs = []
        for techs in df["technologies"]:
            all_techs.extend(techs)
        
        if all_techs:
            top_techs = Counter(all_techs).most_common(5)
            print("\nTop Technologies Discussed:")
            for tech, count in top_techs:
                print(f"  {tech}: {count} articles ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()