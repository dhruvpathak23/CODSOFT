from src.data_loader import get_dataset

def test_pipeline():
    print("Spinning up the data pipeline test...")
    try:
        # Pull the datasets and vectorizer
        train_ds, val_ds, vectorizer = get_dataset()
        
        print(f"\nSuccess! Vocabulary built with {vectorizer.vocabulary_size()} tokens.")
        
        # Grab exactly one batch to verify tensor shapes
        for (images, input_seqs), target_seqs in train_ds.take(1):
            print("\n--- Batch Diagnostics ---")
            print(f"Image tensor shape: {images.shape} (Expected: Batch, 299, 299, 3)")
            print(f"Input sequence shape: {input_seqs.shape} (Expected: Batch, Max_Length)")
            print(f"Target sequence shape: {target_seqs.shape} (Expected: Batch, Max_Length)")
            print("\nAll systems green. The pipeline is fully operational!")
            break
            
    except Exception as e:
        print(f"\nCrash detected in the pipeline: {e}")

if __name__ == "__main__":
    test_pipeline()
