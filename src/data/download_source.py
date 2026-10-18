from datasets import load_dataset


def main():
    dataset = load_dataset("NEUDM/semeval-2014")

    dataset.save_to_disk(
        "data/raw/semeval2014_hf"
    )

    print("Dataset downloaded and saved successfully.")
    print(dataset)


if __name__ == "__main__":
    main()