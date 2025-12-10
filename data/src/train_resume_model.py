
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_PATH = Path("data/resumes.csv")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    return df

def build_pipeline():
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    clf = LogisticRegression(max_iter=200)

    pipe = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )
    return pipe

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "resumes.csv not found. Place it in data/ folder."
        )

    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
