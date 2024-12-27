import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

class URLEnsembleClassifier:
    def __init__(self, bert_model, device='cuda', n_folds=5):
        self.bert_model = bert_model
        self.device = device
        self.n_folds = n_folds
        self.cv_results = {}
        self.scaler = None

        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )

        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1.0
        )

        self.lgb_classifier = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42
        )

        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        self.et_classifier = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )

        self.ada_classifier = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            algorithm='SAMME'
        )

        self.svm_classifier = SVC(
            probability=True,
            kernel='rbf',
            random_state=42
        )

        self.meta_learner = LogisticRegression(max_iter=1000)

        self.models = {
            'bert': self.bert_model,
            'rf': self.rf_classifier,
            'xgb': self.xgb_classifier,
            'lgb': self.lgb_classifier,
            'gb': self.gb_classifier,
            'et': self.et_classifier,
            'ada': self.ada_classifier,
            'svm': self.svm_classifier
        }

        n_models = len(self.models)
        self.model_weights = {model: 1.0/n_models for model in self.models}

        self.model_performance = {model: [] for model in self.models}

        self.feature_importance = {}

        self.fold_metrics = {
            'train_f1': [],
            'val_f1': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': []
        }

    def cross_validate_models(self, features, labels):
        """Perform k-fold cross validation for all models"""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = {model: [] for model in self.models if model != 'bert'}

        print(f"\nPerforming {self.n_folds}-fold cross validation:")

        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            fold_train_preds = []
            fold_val_preds = []

            fold_preds = {}
            for name, model in self.models.items():
                if name != 'bert':  
                    print(f"Training {name}...")
                    model.fit(X_train, y_train)

                    train_probs = model.predict_proba(X_train)
                    val_probs = model.predict_proba(X_val)

                    fold_train_preds.append(train_probs)
                    fold_val_preds.append(val_probs)

                    score = model.score(X_val, y_val)
                    cv_scores[name].append(score)
                    fold_preds[name] = model.predict_proba(X_val)

            train_metrics = self._calculate_fold_metrics(
                np.mean(fold_train_preds, axis=0),
                y_train
            )
            val_metrics = self._calculate_fold_metrics(
                np.mean(fold_val_preds, axis=0),
                y_val
            )

            self.fold_metrics['train_f1'].append(train_metrics['f1'])
            self.fold_metrics['val_f1'].append(val_metrics['f1'])
            self.fold_metrics['train_precision'].append(train_metrics['precision'])
            self.fold_metrics['val_precision'].append(val_metrics['precision'])
            self.fold_metrics['train_recall'].append(train_metrics['recall'])
            self.fold_metrics['val_recall'].append(val_metrics['recall'])

            self._analyze_fold_ensemble(fold_preds, y_val, fold)

        self._summarize_cv_results(cv_scores)

        if hasattr(self, 'visualizer'):
            self.visualizer.plot_cross_validation_metrics(self.fold_metrics)

    def _analyze_fold_ensemble(self, fold_preds, y_true, fold):
        """Analyze ensemble performance for a single fold"""

        weight_schemes = {
            'equal': {model: 1.0/len(fold_preds) for model in fold_preds},
            'weighted': self.model_weights,
            'confidence': self._calculate_confidence_weights(fold_preds)
        }

        for scheme_name, weights in weight_schemes.items():

            ensemble_pred = sum(
                weights[model] * preds 
                for model, preds in fold_preds.items()
            )

            accuracy = np.mean(np.argmax(ensemble_pred, axis=1) == y_true)

            if scheme_name not in self.cv_results:
                self.cv_results[scheme_name] = []
            self.cv_results[scheme_name].append(accuracy)

    def _calculate_confidence_weights(self, predictions):
        """Calculate weights based on prediction confidence"""
        confidences = {}
        for name, preds in predictions.items():
            confidences[name] = np.mean(np.max(preds, axis=1))

        total_conf = sum(confidences.values())
        return {name: conf/total_conf for name, conf in confidences.items()}

    def _summarize_cv_results(self, cv_scores):
        """Print summary of cross-validation results"""
        print("\nCross-validation Results:")

        results_df = pd.DataFrame(cv_scores)
        print("\nIndividual Model Performance:")
        print(f"Mean Accuracy (std):")
        for model in results_df.columns:
            mean = results_df[model].mean()
            std = results_df[model].std()
            print(f"{model}: {mean:.4f} (±{std:.4f})")

        print("\nEnsemble Performance:")
        for scheme, scores in self.cv_results.items():
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"{scheme} weights: {mean:.4f} (±{std:.4f})")

    def train_traditional_models(self, features, labels, bert_probs=None):
        """Modified to include cross-validation and BERT predictions"""

        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        self.cross_validate_models(features, labels)

        print("\nTraining final models on full dataset:")
        train_preds = []

        for name, model in self.models.items():
            if name != 'bert':
                print(f"Training {name}...")
                model.fit(features, labels)
                pred = model.predict_proba(features)
                train_preds.append(pred)

                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_

        print("Obtaining BERT model predictions for meta-learner...")
        if bert_probs is None:

            print("Warning: BERT inputs not provided, using features only")
            bert_probs = np.zeros((len(features), 2))

        train_preds.append(bert_probs)  

        meta_features = np.hstack(train_preds)
        self.meta_learner.fit(meta_features, labels)

        self._print_feature_importance()

    def _print_feature_importance(self):
        """Print aggregated feature importance"""
        if self.feature_importance:
            print("\nFeature Importance Summary:")
            avg_importance = np.zeros_like(next(iter(self.feature_importance.values())))
            for imp in self.feature_importance.values():
                avg_importance += imp
            avg_importance /= len(self.feature_importance)

            for idx in np.argsort(avg_importance)[::-1][:10]:
                print(f"Feature {idx}: {avg_importance[idx]:.4f}")

    def update_weights(self, val_metrics):

        pass  

    def predict(self, bert_logits, features):
        """Combine predictions from all models using the meta-learner"""

        bert_probs = bert_logits.cpu().numpy()

        model_preds = []
        for name, model in self.models.items():
            if name != 'bert':
                try:
                    preds = model.predict_proba(features)
                    model_preds.append(preds)
                except Exception as e:
                    print(f"Error getting predictions from {name}: {e}")
                    continue  

        model_preds.append(bert_probs)

        if model_preds:
            meta_features = np.hstack(model_preds)
        else:
            print("No model predictions available.")
            return bert_probs  

        if hasattr(self, 'meta_learner') and self.meta_learner:
            ensemble_pred = self.meta_learner.predict_proba(meta_features)
        else:

            ensemble_pred = sum(
                self.model_weights[model] * preds 
                for model, preds in zip(self.models.keys(), model_preds)
                if model in self.model_weights
            )
            ensemble_pred /= sum(self.model_weights[model] for model in self.model_weights if model in self.models)

        return ensemble_pred

    def print_meta_learner_coefficients(self):
        """Print meta-learner coefficients"""
        print("\nMeta-Learner Coefficients:")
        feature_names = list(self.models.keys())  
        coefficients = self.meta_learner.coef_[0]
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")

    def _calculate_fold_metrics(self, probs, true_labels):
        """Calculate precision, recall, and F1 score for fold"""
        predictions = np.argmax(probs, axis=1)

        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }