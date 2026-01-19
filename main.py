import pandas as pd
from config import Config
from feature_extractor import FeatureExtractor
from evaluator import Evaluator

def main():
    """Complete pipeline: feature extraction and evaluation."""    
    results = []
    
    for model_name in Config.MODELS.keys():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        extractor = FeatureExtractor(model_name)
        
        print("\nExtracting TRAIN features...")
        train_features, train_positions = extractor.extract_features(Config.TRAIN_SESSIONS)
        extractor.save_features(train_features, train_positions, "train")
        
        print("\nExtracting TEST features...")
        test_features, test_positions = extractor.extract_features(Config.TEST_SESSIONS)
        extractor.save_features(test_features, test_positions, "test")
        
        print("\nEvaluating classification...")
        evaluator = Evaluator(model_name)
        metrics = evaluator.evaluate()
        evaluator.plot_confusion_matrix(metrics["confusion_matrix"])
        
        results.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Feature_Dim": Config.MODELS[model_name]["feature_dim"],
            "Train_Images": len(train_features),
            "Test_Images": len(test_features)
        })
        
        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Accuracy", ascending=False)
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    print(df_results.to_string(index=False))
    
    report_path = Config.REPORTS_DIR / "benchmark_report.csv"
    df_results.to_csv(report_path, index=False)
    print(f"\nReport saved: {report_path}")

if __name__ == "__main__":
    main()