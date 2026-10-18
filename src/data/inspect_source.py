from datasets import load_from_disk


def main():
    dataset = load_from_disk(
        "data/raw/semeval2014_hf"
    )

    print("Dataset structure:")
    print(dataset)

    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Rows: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")
        print(f"Features: {split_data.features}")

        print("\nSample records:")

        sample_count = min(3, len(split_data))

        for index in range(sample_count):
            print(split_data[index])


if __name__ == "__main__":
    main()