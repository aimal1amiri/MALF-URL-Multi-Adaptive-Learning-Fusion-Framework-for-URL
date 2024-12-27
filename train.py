import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import URLDataPreprocessor
from model_setup import URLBertClassifier
import torch
import gc
from visualization import ModelVisualizer
import os
from hyperparameter_tuning import HyperparameterOptimizer
from ensemble_classifier import URLEnsembleClassifier
import numpy as np
import argparse

def train_model(args):

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.optimize_hyperparams:
        print("\nStarting hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(args.dataset_path, n_trials=args.n_trials)
        best_params = optimizer.optimize()
        print("\nBest hyperparameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")

    batch_size = best_params['batch_size'] if args.optimize_hyperparams else args.batch_size

    if args.debug:
        print("\nRunning in debug mode with reduced sample size...")
        preprocessor = URLDataPreprocessor(
            batch_size=batch_size, 
            debug_sample_size=args.debug_sample_size
        )
    else:
        print("\nRunning in production mode...")
        preprocessor = URLDataPreprocessor(batch_size=batch_size)

    train_loader, val_loader = preprocessor.prepare_data(args.dataset_path)

    if args.optimize_hyperparams:
        classifier = URLBertClassifier(
            device=device,
            dropout_rate=best_params['dropout_rate'],
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers']
        )
    else:
        classifier = URLBertClassifier(device=device)

    visualizer = ModelVisualizer()
    save_path = 'model_plots'
    os.makedirs(save_path, exist_ok=True)

    ensemble = URLEnsembleClassifier(classifier, device=device, n_folds=args.n_folds)
    ensemble.visualizer = visualizer

    features = preprocessor.extract_traditional_features(args.dataset_path)
    labels = preprocessor.get_labels(args.dataset_path)

    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)

    ensemble.scaler = preprocessor.scaler

    try:

        bert_probs_list = []

        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_features = batch['features'].to(device)

                bert_outputs = classifier(input_ids, attention_mask, batch_features)
                batch_probs = torch.softmax(bert_outputs, dim=1)
                bert_probs_list.append(batch_probs.cpu().numpy())

                del input_ids, attention_mask, batch_features, bert_outputs, batch_probs
                torch.cuda.empty_cache()

        all_bert_probs = np.vstack(bert_probs_list)

        ensemble.train_traditional_models(features, labels, bert_probs=all_bert_probs)
    except Exception as e:
        print(f"Warning: Could not collect BERT inputs ({str(e)})")
        print("Falling back to traditional features only")
        ensemble.train_traditional_models(features, labels)

    classifier.ensemble = ensemble

    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")

        train_loss = classifier.train_epoch(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss, val_accuracy, metrics = classifier.evaluate(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(classifier.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        visualizer.update_metrics(
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            bert_accuracy=metrics['bert_accuracy'],
            ensemble_accuracy=metrics['ensemble_accuracy'],
            val_precision=metrics['precision'],
            val_recall=metrics['recall'],
            val_f1=metrics['f1']
        )

    visualizer.plot_training_history(save_path)
    visualizer.plot_confusion_matrix(metrics['true_labels'], metrics['predictions'], save_path)
    visualizer.plot_feature_importance(metrics['feature_names'], metrics['importance_scores'], save_path)
    visualizer.plot_learning_dynamics(save_path)

    visualizer.print_performance_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='URL Classification Training Script')

    parser.add_argument('--dataset_path', type=str, default='dataset_50k.csv',
                      help='Path to the dataset CSV file')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode with reduced sample size')
    parser.add_argument('--debug_sample_size', type=int, default=1000,
                      help='Sample size to use in debug mode')

    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--n_folds', type=int, default=5,
                      help='Number of cross-validation folds')
    parser.add_argument('--patience', type=int, default=3,
                      help='Early stopping patience')

    parser.add_argument('--optimize_hyperparams', action='store_true',
                      help='Perform hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                      help='Number of optimization trials')

    args = parser.parse_args()

    train_model(args)