import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
import torch
import torch.nn as nn
import joblib
import json
import glob

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .config import NUM_CHECKPOINTS, MODELS


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

        # Weight initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class TabNetCheckpointCallback(Callback):
    """
    Custom callback to save TabNet model checkpoints at equal intervals
    """
    def __init__(self, save_path, num_checkpoints=NUM_CHECKPOINTS):
        """
        Parameters:
        -----------
        save_path : str
            Base path to save checkpoints (without extension)
        num_checkpoints : int
            Total number of checkpoints to save (including initial and final)
        """
        super().__init__()
        self.save_path = save_path
        self.num_checkpoints = num_checkpoints
        self.checkpoint_epochs = []
        self.current_checkpoint_idx = 0
        
    def on_train_begin(self, logs=None):
        """Calculate checkpoint epochs at the beginning of training"""
        max_epochs = self.trainer.max_epochs
        
        # Calculate equal intervals for checkpoints
        # We want num_checkpoints points including epoch 0 and final epoch
        if self.num_checkpoints <= 1:
            self.checkpoint_epochs = [0]
        else:
            # Create evenly spaced checkpoints
            self.checkpoint_epochs = [
                int(round(i * max_epochs / (self.num_checkpoints - 1)))
                for i in range(self.num_checkpoints)
            ]
            # Ensure uniqueness and sort
            self.checkpoint_epochs = sorted(list(set(self.checkpoint_epochs)))
        
        # Save initial model (epoch 0)
        if 0 in self.checkpoint_epochs:
            checkpoint_path = f"{self.save_path}_checkpoint_0"
            self.trainer.save_model(checkpoint_path)
            print(f"\n  Checkpoint saved: {checkpoint_path}.zip (epoch 0)")
            self.current_checkpoint_idx = 1
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if current epoch matches checkpoint schedule"""
        # epoch is 0-indexed in the callback
        current_epoch = epoch + 1
        
        if (self.current_checkpoint_idx < len(self.checkpoint_epochs) and 
            current_epoch >= self.checkpoint_epochs[self.current_checkpoint_idx]):
            
            checkpoint_path = f"{self.save_path}_checkpoint_{self.current_checkpoint_idx}"
            self.trainer.save_model(checkpoint_path)
            print(f"\n  Checkpoint saved: {checkpoint_path}.zip (epoch {current_epoch})")
            self.current_checkpoint_idx += 1


class KerasMLP:
    """
    Wrapper for Keras MLP to provide sklearn-like interface
    """
    def __init__(self, hidden_layer_sizes=(64, 32), activation='relu', 
                 learning_rate=0.001, batch_size=32, epochs=500, 
                 validation_split=0.1, early_stopping_patience=20,
                 random_state=42):
        """
        Initialize Keras MLP classifier
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple
            Sizes of hidden layers
        activation : str
            Activation function
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs
        validation_split : float
            Fraction of training data to use as validation
        early_stopping_patience : int
            Patience for early stopping
        random_state : int
            Random seed
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model = None
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_model(self, n_features, n_classes):
        """Build Keras model"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=(n_features,)))
        
        # Hidden layers
        for units in self.hidden_layer_sizes:
            model.add(layers.Dense(units, activation=self.activation))
            model.add(layers.Dropout(0.2))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
        
        Returns:
        --------
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Build model
        self.model = self._build_model(self.n_features_in_, self.n_classes_)
        
        # Prepare callbacks
        callback_list = []
        if self.validation_split > 0 and self.early_stopping_patience > 0:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
            callback_list.append(early_stop)
        
        # Train model
        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            sample_weight=sample_weight,
            callbacks=callback_list,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        proba = self.predict_proba(X)
        
        if self.n_classes_ == 2:
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Predicted probabilities
        """
        X = np.array(X)
        
        if self.n_classes_ == 2:
            proba_pos = self.model.predict(X, verbose=0).flatten()
            proba = np.vstack([1 - proba_pos, proba_pos]).T
        else:
            proba = self.model.predict(X, verbose=0)
        
        return proba
    
    def save(self, filepath):
        """Save model to .keras file"""
        # Save Keras model using new .keras format
        self.model.save(f"{filepath}.keras")
        
        # Save metadata
        metadata = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'n_features_in_': self.n_features_in_,
            'classes_': self.classes_.tolist(),
            'n_classes_': self.n_classes_
        }
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from .keras file"""
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            hidden_layer_sizes=tuple(metadata['hidden_layer_sizes']),
            activation=metadata['activation'],
            learning_rate=metadata['learning_rate'],
            batch_size=metadata['batch_size'],
            epochs=metadata['epochs'],
            validation_split=metadata['validation_split'],
            early_stopping_patience=metadata['early_stopping_patience'],
            random_state=metadata['random_state']
        )
        
        # Load Keras model using new .keras format
        instance.model = keras.models.load_model(f"{filepath}.keras")
        instance.n_features_in_ = metadata['n_features_in_']
        instance.classes_ = np.array(metadata['classes_'])
        instance.n_classes_ = metadata['n_classes_']
        
        return instance


def get_classifier(classifier_name, n_samples=None, n_features=None, random_state=42, **kwargs):
    """
    Get classifier instance with adaptive parameter selection based on data scale
    
    Parameters:
    -----------
    classifier_name : str
        Classifier name, supports: 'LR', 'SVM', 'DT', 'RF', 'MLP', 'XGBoost', 'TabNet'
        Case insensitive
    n_samples : int, optional
        Number of training samples for adaptive parameter adjustment
    n_features : int, optional
        Number of features for adaptive parameter adjustment
    random_state : int
        Random seed, default 42
    **kwargs : dict
        Additional model parameters that will override default parameters
    
    Returns:
    --------
    classifier : classifier instance
        Supports .fit(), .predict(), .predict_proba() methods
    
    Examples:
    ---------
    >>> # Automatically adjust parameters based on data scale
    >>> clf = get_classifier('TabNet', n_samples=500, n_features=8)
    >>> clf.fit(X_train, y_train)
    >>> 
    >>> # Manually override parameters
    >>> clf = get_classifier('XGBoost', n_samples=5000, n_estimators=200)
    """
    
    classifier_name = classifier_name.upper()
    
    # ==================== Logistic Regression ====================
    if classifier_name == 'LR':
        default_params = {
            'max_iter': 1000,
            'random_state': random_state,
            'solver': 'lbfgs',
            'C': 1.0
        }
        
        if n_samples is not None and n_samples < 1000:
            default_params['C'] = 0.1
        
        default_params.update(kwargs)
        return LogisticRegression(**default_params)
    
    # ==================== Support Vector Machine ====================
    elif classifier_name == 'SVM':
        default_params = {
            'kernel': 'rbf',
            'probability': True,
            'random_state': random_state,
            'max_iter': -1,
            'C': 1.0,
            'gamma': 'scale',
            'tol': 1e-4,
            'cache_size': 2000
        }
        
        if n_samples is not None:
            if n_samples < 500:
                default_params['C'] = 0.5
                default_params['gamma'] = 'auto'
            elif n_samples < 2000:
                default_params['C'] = 0.8
            elif n_samples >= 10000:
                print(f"Large dataset ({n_samples} samples), using LinearSVC instead of SVC")
                from sklearn.svm import LinearSVC
                from sklearn.calibration import CalibratedClassifierCV
                
                base_svm = LinearSVC(
                    max_iter=10000,
                    random_state=random_state,
                    C=0.1,
                    tol=1e-4,
                    dual='auto'
                )
                
                return CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
        
        default_params.update(kwargs)
        return SVC(**default_params)
    
    # ==================== Decision Tree ====================
    elif classifier_name == 'DT':
        default_params = {
            'random_state': random_state,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
        
        if n_samples is not None:
            if n_samples < 500:
                default_params['max_depth'] = 5
                default_params['min_samples_split'] = 20
                default_params['min_samples_leaf'] = 10
            elif n_samples < 2000:
                default_params['max_depth'] = 7
                default_params['min_samples_split'] = 15
                default_params['min_samples_leaf'] = 8
            elif n_samples >= 10000:
                default_params['max_depth'] = 15
                default_params['min_samples_split'] = 5
                default_params['min_samples_leaf'] = 2
        
        default_params.update(kwargs)
        return DecisionTreeClassifier(**default_params)
    
    # ==================== Random Forest ====================
    elif classifier_name == 'RF':
        default_params = {
            'random_state': random_state,
            'n_jobs': -1,
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
        
        if n_samples is not None:
            if n_samples < 500:
                default_params['n_estimators'] = 50
                default_params['max_depth'] = 5
                default_params['min_samples_split'] = 20
                default_params['min_samples_leaf'] = 10
            elif n_samples < 2000:
                default_params['n_estimators'] = 80
                default_params['max_depth'] = 8
                default_params['min_samples_split'] = 15
                default_params['min_samples_leaf'] = 8
            elif n_samples >= 10000:
                default_params['n_estimators'] = 150
                default_params['max_depth'] = 15
                default_params['min_samples_split'] = 5
                default_params['min_samples_leaf'] = 2
        
        default_params.update(kwargs)
        return RandomForestClassifier(**default_params)
    
    # ==================== Keras MLP ====================
    elif classifier_name == 'MLP':
        default_params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 500,
            'validation_split': 0.1,
            'early_stopping_patience': 20,
            'random_state': random_state
        }
        
        if n_samples is not None:
            if n_samples < 500:
                default_params['hidden_layer_sizes'] = (32, 16)
                default_params['learning_rate'] = 0.0005
                default_params['epochs'] = 300
                default_params['validation_split'] = 0.0
                default_params['early_stopping_patience'] = 0
                default_params['batch_size'] = 16
            elif n_samples < 2000:
                default_params['hidden_layer_sizes'] = (48, 24)
                default_params['learning_rate'] = 0.0008
                default_params['epochs'] = 400
                default_params['batch_size'] = 32
            elif n_samples >= 10000:
                default_params['hidden_layer_sizes'] = (128, 64)
                default_params['learning_rate'] = 0.001
                default_params['epochs'] = 600
                default_params['batch_size'] = 64
        
        default_params.update(kwargs)
        return KerasMLP(**default_params)
    
    # ==================== XGBoost ====================
    elif classifier_name == 'XGBOOST':
        default_params = {
            'random_state': random_state,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'tree_method': 'hist',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
        
        if n_samples is not None:
            if n_samples < 500:
                default_params['n_estimators'] = 50
                default_params['max_depth'] = 3
                default_params['learning_rate'] = 0.05
                default_params['min_child_weight'] = 3
                default_params['reg_alpha'] = 0.1
                default_params['reg_lambda'] = 2
                default_params['subsample'] = 0.7
                default_params['colsample_bytree'] = 0.7
            elif n_samples < 2000:
                default_params['n_estimators'] = 80
                default_params['max_depth'] = 4
                default_params['learning_rate'] = 0.08
                default_params['min_child_weight'] = 2
                default_params['reg_lambda'] = 1.5
            elif n_samples >= 10000:
                default_params['n_estimators'] = 150
                default_params['max_depth'] = 8
                default_params['learning_rate'] = 0.1
                default_params['min_child_weight'] = 1
        
        if n_features is not None and n_features <= 8:
            default_params['colsample_bytree'] = 1.0
        
        default_params.update(kwargs)
        return XGBClassifier(**default_params)
    
    # ==================== TabNet ====================
    elif classifier_name == 'TABNET':
        default_params = {
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'scheduler_params': dict(step_size=10, gamma=0.9),
            'mask_type': 'sparsemax',
            'seed': random_state,
            'verbose': 0,
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'lambda_sparse': 1e-3,
            'momentum': 0.02,
            'n_independent': 2,
            'n_shared': 2
        }
        
        if n_samples is not None:
            if n_samples < 500:
                default_params.update({
                    'n_d': 4,
                    'n_a': 4,
                    'n_steps': 3,
                    'gamma': 1.2,
                    'lambda_sparse': 5e-3,
                    'n_independent': 1,
                    'n_shared': 1,
                    'momentum': 0.05
                })
                
            elif n_samples < 2000:
                default_params.update({
                    'n_d': 6,
                    'n_a': 6,
                    'n_steps': 3,
                    'gamma': 1.3,
                    'lambda_sparse': 3e-3,
                    'n_independent': 1,
                    'n_shared': 2
                })
                
            elif n_samples < 10000:
                default_params.update({
                    'n_d': 8,
                    'n_a': 8,
                    'n_steps': 4,
                    'gamma': 1.5,
                    'lambda_sparse': 1e-3,
                    'n_independent': 2,
                    'n_shared': 2
                })
                
            else:
                default_params.update({
                    'n_d': 16,
                    'n_a': 16,
                    'n_steps': 5,
                    'gamma': 1.5,
                    'lambda_sparse': 5e-4,
                    'n_independent': 2,
                    'n_shared': 3
                })
        
        if n_features is not None and n_features <= 6:
            current_n_d = default_params['n_d']
            default_params['n_d'] = max(4, current_n_d // 2)
            default_params['n_a'] = max(4, current_n_d // 2)
            default_params['n_steps'] = min(3, default_params['n_steps'])
        
        default_params.update(kwargs)
        return TabNetClassifier(**default_params)
    
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}. "
                        f"Supported options: LR, SVM, DT, RF, MLP, XGBoost, TabNet")


def get_tabnet_fit_params(n_samples, **kwargs):
    """
    Return TabNet fit() parameters based on sample size
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples
    **kwargs : dict
        Additional parameters that will override defaults
    
    Returns:
    --------
    fit_params : dict
        Parameter dictionary to pass to clf.fit()
    
    Examples:
    ---------
    >>> fit_params = get_tabnet_fit_params(n_samples=500)
    >>> clf.fit(X_train, y_train, **fit_params)
    """
    
    fit_params = {
        'max_epochs': 100,
        'patience': 15,
        'batch_size': 128,
        'virtual_batch_size': 64
    }
    
    if n_samples < 500:
        fit_params.update({
            'max_epochs': 200,
            'patience': 30,
            'batch_size': 32,
            'virtual_batch_size': 16
        })
        
    elif n_samples < 2000:
        fit_params.update({
            'max_epochs': 150,
            'patience': 20,
            'batch_size': 64,
            'virtual_batch_size': 32
        })
        
    elif n_samples >= 10000:
        fit_params.update({
            'max_epochs': 100,
            'patience': 10,
            'batch_size': 256,
            'virtual_batch_size': 128
        })
    
    fit_params.update(kwargs)
    
    return fit_params


def save_classifier(clf, model_name, save_path):
    """
    Save classifier with appropriate method based on model type
    
    Parameters:
    -----------
    clf : classifier instance
        Trained classifier
    model_name : str
        Model name (e.g., 'LR', 'XGBoost', 'TabNet', 'MLP')
    save_path : str
        Path to save the model (without extension)
    """
    model_name = model_name.upper()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    if model_name == 'TABNET':
        clf.save_model(save_path)
        print(f"TabNet model saved to {save_path}.zip")
        
    elif model_name == 'XGBOOST':
        clf.save_model(f"{save_path}.json")
        print(f"XGBoost model saved to {save_path}.json")
        
    elif model_name == 'MLP':
        clf.save(save_path)
        print(f"Keras MLP model saved to {save_path}.keras and {save_path}_metadata.json")
        
    else:
        joblib.dump(clf, f"{save_path}.pkl")
        print(f"{model_name} model saved to {save_path}.pkl")


def load_maat_ensemble(model_name, load_path, checkpoint_idx=None):
    """
    Load MAAT ensemble models (multiple fairness models + one performance model)
    
    Parameters:
    -----------
    model_name : str
        Model name (e.g., 'LR', 'TabNet')
    load_path : str
        Base path to the model directory (e.g., 'models/census/maat/lr/seed_0/lr')
    checkpoint_idx : int, optional
        For TabNet only: load specific checkpoint
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'fairness_models': list of fairness model instances
        - 'performance_model': performance model instance
        - 'n_fairness_models': number of fairness models
    
    Examples:
    ---------
    >>> models = load_maat_ensemble('TabNet', 'models/census/maat/tabnet/seed_0/tabnet')
    >>> fairness_models = models['fairness_models']
    >>> performance_model = models['performance_model']
    """
    base_dir = os.path.dirname(load_path)
    model_basename = os.path.basename(load_path)
    
    # Load ensemble info to get number of fairness models
    ensemble_info_path = os.path.join(base_dir, f"{model_basename}_ensemble_info.json")
    
    if os.path.exists(ensemble_info_path):
        with open(ensemble_info_path, 'r') as f:
            ensemble_info = json.load(f)
        n_fairness_models = ensemble_info['n_fairness_models']
    else:
        # Fallback: detect from file system
        if model_name.upper() == 'TABNET':
            pattern = f"{load_path}_fairness_*.zip"
        elif model_name.upper() == 'XGBOOST':
            pattern = f"{load_path}_fairness_*.json"
        elif model_name.upper() == 'MLP':
            pattern = f"{load_path}_fairness_*.keras"
        else:
            pattern = f"{load_path}_fairness_*.pkl"
        
        fairness_files = glob.glob(pattern)
        n_fairness_models = len(fairness_files)
        
        if n_fairness_models == 0:
            raise FileNotFoundError(f"No fairness models found at {load_path}")
    
    print(f"Loading MAAT ensemble: {n_fairness_models} fairness models + 1 performance model")
    
    # Load fairness models
    fairness_models = []
    for i in range(n_fairness_models):
        fairness_path = f"{load_path}_fairness_{i}"
        fairness_clf = load_classifier(model_name, fairness_path, checkpoint_idx, method_name=None)
        fairness_models.append(fairness_clf)
        print(f"  Loaded fairness model {i+1}/{n_fairness_models}")
    
    # Load performance model
    performance_path = f"{load_path}_performance"
    performance_clf = load_classifier(model_name, performance_path, checkpoint_idx, method_name=None)
    print(f"  Loaded performance model")
    
    return {
        'fairness_models': fairness_models,
        'performance_model': performance_clf,
        'n_fairness_models': n_fairness_models
    }


def load_classifier(model_name, load_path, checkpoint_idx=None, method_name=None):
    """
    Load classifier with appropriate method based on model type
    
    Parameters:
    -----------
    model_name : str
        Model name (e.g., 'LR', 'XGBoost', 'TabNet', 'MLP')
    load_path : str
        Path to load the model (without extension for TabNet/XGBoost/MLP)
        For MAAT: this should be the directory containing all models
    checkpoint_idx : int, optional
        For TabNet only: load specific checkpoint (0-9)
        If None, load the final model
    method_name : str, optional
        If 'maat', returns a dictionary of models instead of a single model
    
    Returns:
    --------
    clf : classifier instance or dict
        - Single classifier for normal methods
        - Dict with 'fairness_models' and 'performance_model' for MAAT
    
    Examples:
    ---------
    >>> # Load normal model
    >>> clf = load_classifier('TabNet', 'models/tabnet_model')
    >>> 
    >>> # Load MAAT ensemble
    >>> models = load_classifier('TabNet', 'models/seed_0/tabnet', method_name='maat')
    >>> # Returns: {'fairness_models': [clf1, clf2, ...], 'performance_model': clf_perf}
    """
    model_name = model_name.upper()
    
    # ==================== MAAT Special Handling ====================
    if method_name == 'maat':
        return load_maat_ensemble(model_name, load_path, checkpoint_idx)
    
    # ==================== Normal Single Model Loading ====================
    if model_name == 'TABNET':
        clf = TabNetClassifier()
        
        if checkpoint_idx is not None:
            load_path = f"{load_path}_checkpoint_{checkpoint_idx}"
            if not load_path.endswith('.zip'):
                load_path = f"{load_path}.zip"
            clf.load_model(load_path)
            print(f"TabNet checkpoint {checkpoint_idx} loaded from {load_path}")
        else:
            if not load_path.endswith('.zip'):
                load_path = f"{load_path}.zip"
            clf.load_model(load_path)
            print(f"TabNet model loaded from {load_path}")
        
        return clf
        
    elif model_name == 'XGBOOST':
        if checkpoint_idx is not None:
            print(f"Warning: checkpoint_idx is only supported for TabNet, ignoring for {model_name}")
        
        clf = XGBClassifier()
        if not load_path.endswith('.json'):
            load_path = f"{load_path}.json"
        clf.load_model(load_path)
        print(f"XGBoost model loaded from {load_path}")
        return clf
        
    elif model_name == 'MLP':
        if checkpoint_idx is not None:
            print(f"Warning: checkpoint_idx is only supported for TabNet, ignoring for {model_name}")
        
        clf = KerasMLP.load(load_path)
        print(f"Keras MLP model loaded from {load_path}")
        return clf
        
    else:
        if checkpoint_idx is not None:
            print(f"Warning: checkpoint_idx is only supported for TabNet, ignoring for {model_name}")
        
        if not load_path.endswith('.pkl'):
            load_path = f"{load_path}.pkl"
        clf = joblib.load(load_path)
        print(f"{model_name} model loaded from {load_path}")
        return clf


def fit_classifier(clf, model_name, X_train, y_train, sample_weight=None, 
                   save_checkpoints=False, checkpoint_path=None, **fit_kwargs):
    """
    Unified training interface with sample_weight support
    
    Parameters:
    -----------
    clf : classifier instance
        Classifier to train
    model_name : str
        Model name
    X_train, y_train : array-like
        Training data
    sample_weight : array-like, optional
        Sample weights (now supported by all models including MLP)
    save_checkpoints : bool, optional
        For TabNet only: whether to save 10 checkpoints during training
        Default is False (old behavior)
    checkpoint_path : str, optional
        For TabNet only: base path to save checkpoints (without extension)
        Required if save_checkpoints=True
        Note: You still need to manually call save_classifier() to save the final model
    **fit_kwargs : dict
        Additional fit parameters (e.g., for TabNet)
    
    Returns:
    --------
    clf : trained classifier
    
    Examples:
    ---------
    >>> # Default behavior (no checkpoints)
    >>> clf = fit_classifier(clf, 'TabNet', X_train, y_train)
    >>> save_classifier(clf, 'TabNet', 'models/tabnet_model')
    >>> 
    >>> # Save 10 checkpoints during training
    >>> clf = fit_classifier(clf, 'TabNet', X_train, y_train, 
    ...                      save_checkpoints=True, 
    ...                      checkpoint_path='models/tabnet_model')
    >>> # Checkpoints are saved automatically, but you still need to save final model:
    >>> save_classifier(clf, 'TabNet', 'models/tabnet_model')
    """
    model_name = model_name.upper()
    
    if model_name == 'TABNET':
        # TabNet uses 'weights' instead of 'sample_weight'
        if sample_weight is not None:
            fit_kwargs['weights'] = sample_weight
        
        # Add default TabNet fit parameters if not provided
        if 'max_epochs' not in fit_kwargs:
            tabnet_params = get_tabnet_fit_params(len(X_train))
            fit_kwargs.update(tabnet_params)
        
        # Handle checkpoint saving
        if save_checkpoints:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided when save_checkpoints=True")
            
            # Create checkpoint callback
            checkpoint_callback = TabNetCheckpointCallback(
                save_path=checkpoint_path,
                num_checkpoints=NUM_CHECKPOINTS
            )
            
            # Add callback to fit_kwargs
            if 'callbacks' not in fit_kwargs:
                fit_kwargs['callbacks'] = []
            fit_kwargs['callbacks'].append(checkpoint_callback)
            
            print(f"Training TabNet with checkpoint saving enabled...")
            print(f"Checkpoints will be saved to: {checkpoint_path}_checkpoint_*.zip")
            print(f"Note: You still need to manually save the final model using save_classifier()")
        
        # Train model
        clf.fit(X_train, y_train, **fit_kwargs)
        
    else:
        # All other models (LR, SVM, DT, RF, MLP, XGBoost) support 'sample_weight'
        if save_checkpoints:
            print(f"Warning: save_checkpoints is only supported for TabNet, ignoring for {model_name}")
        
        if sample_weight is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        else:
            clf.fit(X_train, y_train, **fit_kwargs)
    
    return clf


def compute_tabnet_checkpoint_importances(test_X, checkpoint_base_path, num_checkpoints=NUM_CHECKPOINTS):
    """
    Load TabNet checkpoints and compute global feature importance for each checkpoint
    
    Parameters:
    -----------
    test_X : array-like, shape (n_samples, n_features)
        Test samples for computing feature importance
    checkpoint_base_path : str
        Base path where checkpoints are saved (without _checkpoint_X suffix)
        Example: 'models/tabnet_model' will load from:
            - models/tabnet_model_checkpoint_0.zip
            - models/tabnet_model_checkpoint_1.zip
            - ...
            - models/tabnet_model_checkpoint_9.zip
    num_checkpoints : int, optional
        Number of checkpoints to load (default: NUM_CHECKPOINTS)
    
    Returns:
    --------
    importance_matrix : numpy.ndarray, shape (n_features, num_checkpoints)
        Feature importance matrix where:
        - Each row corresponds to a feature
        - Each column corresponds to a checkpoint model
        - Values are normalized (each column sums to 1)
    
    Notes:
    ------
    - The function uses TabNet's explain() method to compute feature importance
    - explain() returns a tuple: (importances_array, masks_dict)
    - Global importance is calculated by averaging local importances across all samples
    - The result is automatically saved to: checkpoint_base_path + '_feature_importances.npy'
    """
    
    # Ensure correct data type
    test_X = np.array(test_X, dtype=np.float32)
    n_features = test_X.shape[1]
    
    # Initialize importance matrix: rows=features, columns=checkpoints
    importance_matrix = np.zeros((n_features, num_checkpoints))
    
    print(f"\n{'='*70}")
    print(f"Computing TabNet Feature Importance Across {num_checkpoints} Checkpoints")
    print(f"{'='*70}")
    print(f"Test samples: {test_X.shape[0]}, Features: {n_features}")
    print(f"Base path: {checkpoint_base_path}")
    print(f"{'='*70}\n")
    
    # Load each checkpoint and compute importance
    for checkpoint_idx in range(num_checkpoints):
        try:
            # Load checkpoint using existing function
            clf = load_classifier('TabNet', checkpoint_base_path, checkpoint_idx=checkpoint_idx)
            
            # importances: shape (n_samples, n_features)
            # masks_dict: dict containing attention masks for each step
            result = clf.explain(test_X, normalize=True)
            
            if isinstance(result, tuple):
                local_importances = result[0]
            else:
                local_importances = result
            
            expected_shape = (test_X.shape[0], n_features)
            if local_importances.shape != expected_shape:
                raise ValueError(
                    f"Unexpected importance shape: {local_importances.shape}, "
                    f"expected {expected_shape}"
                )
            
            # Compute global importance by averaging across all samples
            # This follows the paper: Global_Importance = mean_b M_agg(b)
            global_importance = local_importances.mean(axis=0)
            
            # Ensure normalization (should already be close to 1, but ensure precision)
            importance_sum = global_importance.sum()
            if importance_sum > 0:
                global_importance = global_importance / importance_sum
            else:
                print(f"⚠ Warning: Checkpoint {checkpoint_idx} has zero importance sum!")
                global_importance = np.ones(n_features) / n_features
            
            # Store in matrix (column = checkpoint)
            importance_matrix[:, checkpoint_idx] = global_importance
            
            print(f"✓ Checkpoint {checkpoint_idx}: Importance computed "
                  f"(sum={global_importance.sum():.6f}, "
                  f"max={global_importance.max():.4f}, "
                  f"min={global_importance.min():.4f})")
            
        except Exception as e:
            import traceback
            print(f"\n{'!'*70}")
            print(f"✗ Checkpoint {checkpoint_idx}: FAILED")
            print(f"{'!'*70}")
            print(f"Error: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            print(f"{'!'*70}\n")
            
            # Fill with NaN if checkpoint fails
            importance_matrix[:, checkpoint_idx] = np.nan
    
    # Check if all checkpoints failed
    if np.isnan(importance_matrix).all():
        print("\n⚠ WARNING: All checkpoints failed! Returning NaN matrix.")
    elif np.isnan(importance_matrix).any():
        failed_checkpoints = np.where(np.isnan(importance_matrix[0, :]))[0]
        print(f"\n⚠ WARNING: {len(failed_checkpoints)} checkpoint(s) failed: {failed_checkpoints}")
    
    # Save the importance matrix
    save_path = f"{checkpoint_base_path}_feature_importances.npy"
    np.save(save_path, importance_matrix)
    
    print(f"\n{'='*70}")
    print(f"✓ Feature importance matrix saved to: {save_path}")
    print(f"  Matrix shape: {importance_matrix.shape}")
    print(f"  (rows=features, columns=checkpoints)")
    print(f"  Valid checkpoints: {(~np.isnan(importance_matrix[0, :])).sum()}/{num_checkpoints}")
    print(f"{'='*70}\n")
    
    return importance_matrix


def model_standardize_maat(classifier_name, maat_models, scaler):
    """
    Standardize MAAT ensemble models to a unified interface
    
    Parameters:
    -----------
    classifier_name : str
        Classifier name
    maat_models : dict
        Dictionary containing 'fairness_models' and 'performance_model'
    scaler : sklearn scaler
        Feature scaler
    
    Returns:
    --------
    ensemble_func : callable
        Function that takes X and returns ensemble predictions
    reward_type : str
        Type of reward/output
    """
    
    fairness_models = maat_models['fairness_models']
    performance_model = maat_models['performance_model']
    
    def get_maat_ensemble_output(X):
        """
        MAAT ensemble prediction: average of all fairness models + performance model
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = scaler.transform(X)
        
        # Collect predictions from all models
        all_probas = []
        
        # Fairness models
        for fairness_model in fairness_models:
            proba = fairness_model.predict_proba(X_scaled)[:, 1]
            all_probas.append(proba)
        
        # Performance model
        perf_proba = performance_model.predict_proba(X_scaled)[:, 1]
        all_probas.append(perf_proba)
        
        # Average ensemble (default 0.5-0.5 strategy from maat paper)
        all_probas = np.column_stack(all_probas)
        ensemble_proba = np.mean(all_probas, axis=1)
        
        return ensemble_proba.reshape(-1, 1)
    
    if classifier_name in MODELS:
        return get_maat_ensemble_output, "positive_probability"
    else:
        raise NotImplementedError(f"{classifier_name} has not been implemented.")


def model_standardize(classifier_name, orig_model, scaler, method_name=None, sensitive_indices=None):
    """
    Standardize model output to a unified interface
    
    Parameters:
    -----------
    classifier_name : str
        Classifier name
    orig_model : classifier instance or dict
        - Single model for normal methods
        - Dict with 'fairness_models' and 'performance_model' for MAAT
    scaler : sklearn scaler
        Feature scaler
    method_name : str, optional
        Method name (e.g., 'blindness', 'maat')
    sensitive_indices : list, optional
        Indices of sensitive attributes
    
    Returns:
    --------
    model_func : callable
        Function that takes X and returns predictions
    reward_type : str
        Type of reward/output
    """
    
    # ==================== MAAT Special Handling ====================
    if method_name == 'maat':
        return model_standardize_maat(classifier_name, orig_model, scaler)
    
    # ==================== Normal Single Model ====================
    def get_model_output(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if method_name == "blindness":
            for idx in sorted(sensitive_indices):
                X = np.insert(X, idx, 0, axis=1)
            mask = np.ones(X.shape[1], dtype=bool)
            mask[sensitive_indices] = False
            X_scaled = scaler.transform(X)[:, mask]
        else:
            X_scaled = scaler.transform(X)
        return orig_model.predict_proba(X_scaled)[:, 1].reshape(-1, 1)
            
    if classifier_name in MODELS:
        return get_model_output, "positive_probability"
    else:
        raise NotImplementedError(f"{classifier_name} has not been implemented.")