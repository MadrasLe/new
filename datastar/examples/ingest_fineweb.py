from datastar import DataStarIngest

def main():
    # Example usage of DataStarIngest
    # Note: Requires 'datasets' and 'transformers' installed

    ingest = DataStarIngest(
        tokenizer_name="bert-base-uncased", # Smaller tokenizer for demo
        batch_size=1024,
    )

    # Use a small subset or stream
    source = ingest.stream_from_huggingface(
        repo_id="HuggingFaceFW/fineweb-edu",
        split="train",
    )

    # Run for a small amount of tokens as a demo
    ingest.run_pipeline(
        source_iterator=source,
        output_path="fineweb_demo.jsonl",
        quota_tokens=100_000, # Small quota for demo
    )

if __name__ == "__main__":
    main()
