from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def check_dataset_exists():
    try:
        load_dataset("parquet", data_files="datasets/cnn_dailymail_train.parquet", split="train")
        load_dataset("parquet", data_files="datasets/cnn_dailymail_test.parquet", split="train")
        return True
    except FileNotFoundError:
        return False

if not check_dataset_exists():
    print("Tokenized dataset does not exist locally... Generating and saving to disk.")

    def tokenize_and_chunk(dataset, tokenizer, chunk_size=512, train_rows=100_000, test_rows=500):
        buffer = []
        row_count = 0

        for example in dataset:
            tokens = tokenizer(example["article"], truncation=False, padding=False)['input_ids']
            buffer.extend(tokens)

            while len(buffer) >= chunk_size + 1:
                if row_count >= (train_rows + test_rows):
                    return
            
                input_chunk = buffer[:chunk_size]
                target_chunk = buffer[1:chunk_size + 1]
            
                split = "train" if row_count < train_rows else "test"
            
                yield{
                    "split": split,
                    "input": input_chunk,
                    "target": target_chunk
                }
            
                buffer = buffer[chunk_size:]
                row_count += 1

    TRAIN_ROWS = 1400000
    TEST_ROWS = 500
    CHUNK_SIZE = 128

    tokenized_ds = Dataset.from_generator(lambda: tokenize_and_chunk(ds, hf_tokenizer,chunk_size=CHUNK_SIZE, train_rows=TRAIN_ROWS, test_rows=TEST_ROWS))

    data_splits = tokenized_ds.train_test_split(test_size=TEST_ROWS / (TRAIN_ROWS + TEST_ROWS), seed=67)

    data_splits["train"].to_parquet("cnn_dailymail_train.parquet")
    data_splits["test"].to_parquet("cnn_dailymail_test.parquet")
    print(f"✅ Saved {TRAIN_ROWS} train rows and {TEST_ROWS} test rows.")
else:
    print("Tokenized dataset already exists locally.")