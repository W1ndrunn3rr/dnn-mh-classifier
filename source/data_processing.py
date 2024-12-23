from .feature_extractor import FeatureExtractor
import pandas as pd


def main():
    i: int = 0
    df = pd.read_csv("../data/datadata_filtered.csv")
    extractor = FeatureExtractor()
    processed_data = []

    for _, row in df.iterrows():
        try:

            if i > 100000:
                break

            extractor.process_text(row["lyric"])
            features = extractor.process_data()

            if features is None:
                continue

            # Create a list of features and append the label
            processed_data.append(
                {
                    "neg_score": features[0],
                    "neu_score": features[1],
                    "pos_score": features[2],
                    "compound_score": features[3],
                    "n_sentences": features[4],
                    "n_tokens": features[5],
                    "unique_tokens_r": features[6],
                    "nouns_r": features[7],
                    "proper_nouns_r": features[8],
                    "verbs_r": features[9],
                    "adverbs_r": features[10],
                    "adjcetives_r": features[11],
                    "song_tone": features[12],
                    "mean_sentence_length": features[13],
                    "label": row["disorder"],
                }
            )
            i += 1
            print(f"Total progress: {i/100000 * 100}%")

        except Exception:
            print("Error")
            continue

    processed_df = pd.DataFrame(processed_data)

    # Save to CSV
    processed_df.to_csv("processed_data.csv", index=False)
    print("Data saved!")


if __name__ == "__main__":
    main()
