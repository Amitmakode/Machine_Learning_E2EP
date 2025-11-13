from laptop_price.pipeline.training_pipeline import run_pipeline

if __name__ == "__main__":
    summary = run_pipeline()
    print("Pipeline completed. Summary:")
    print(summary)
