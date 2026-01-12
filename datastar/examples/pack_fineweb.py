import os
from datastar import DataStarPacker

def main():
    if not os.path.exists("fineweb_demo.jsonl"):
        print("Run ingest_fineweb.py first to generate the data.")
        return

    packer = DataStarPacker(
        txt_path="fineweb_demo.jsonl",
        bin_path="train_data.bin",
        tokenizer_name="bert-base-uncased",
    )

    packer.run()

if __name__ == "__main__":
    main()
