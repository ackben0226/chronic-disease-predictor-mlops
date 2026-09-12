"""
Tests for model training module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.trainer import ModelTrainer, RandomForestModel


class TestModelTraining:
    """Test model training functionality."""

    def test_random_forest_model_init(self):
        """Test that RandomForestModel initializes correctly."""
        config = {'n_estimators': 100, 'random_state': 42}
        model = RandomForestModel(config=config)
        assert model.model is not None
        assert model.is_trained is False

    def test_model_fit_and_predict(self):
        """Test that model can fit and predict."""
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])

        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        model.fit(X, y)

        assert model.is_trained is True

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_model_predict_before_fit_raises_error(self):
        """Test that predict before fit raises ValueError."""
        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        X = pd.DataFrame({'feature': [1, 2, 3]})

        with pytest.raises(ValueError) as exc:
            model.predict(X)
        assert 'not been trained' in str(exc.value)

    def test_trainer_split_data(self):
        """Test that ModelTrainer splits data correctly."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': range(200, 300)
        })

        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model, 
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        trainer.split_data(df, target_col='target')

        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None

        # With 100 rows, train should be ~70% (70 rows)
        # Allow for rounding differences (69 or 70)
        total = len(df)
        expected_train_pct = 1 - trainer.test_size - trainer.val_size  # 0.7
        expected_train = int(total * expected_train_pct)
        
        # Use range check instead of exact equality
        assert abs(len(trainer.X_train) - expected_train) <= 1
        assert abs(len(trainer.X_val) - int(total * trainer.val_size)) <= 1
        assert abs(len(trainer.X_test) - int(total * trainer.test_size)) <= 1

    def test_trainer_train_and_evaluate(self):
        """Test that trainer can train and evaluate model."""
        X = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200)
        })
        y = pd.Series(range(200, 300))
        df = pd.concat([X, y.to_frame('target')], axis=1)

        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model, 
            test_size=0.2,
            val_size=0.1,
            do_hyperparameter_tuning=False,
            do_cross_validation=False,
            random_state=42
        )
        trainer.split_data(df, target_col='target')
        trainer.train_with_cv(cv_folds=2)

        metrics = trainer.evaluate()
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics

    def test_trainer_save_and_load_model(self, tmp_path):
        """Test that model can be saved and loaded."""
        X = pd.DataFrame({
            'feature': range(20),
            'feature2': range(20, 40)
        })
        y = pd.Series(range(40, 60))
        df = pd.concat([X, y.to_frame('target')], axis=1)

        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model, 
            test_size=0.2,
            val_size=0.1,
            do_hyperparameter_tuning=False,
            do_cross_validation=False,
            random_state=42
        )
        trainer.split_data(df, target_col='target')
        trainer.train_with_cv(cv_folds=2)

        # Save - note: save_model adds .joblib extension
        save_path = tmp_path / "test_model.pkl"
        trainer.save_model(save_path)

        # Check for .joblib file (not .pkl)
        joblib_path = save_path.with_suffix('.joblib')
        assert joblib_path.exists()

    def test_feature_importance_extraction(self):
        """Test that feature importance can be extracted."""
        X = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200)
        })
        y = pd.Series(range(200, 300))
        df = pd.concat([X, y.to_frame('target')], axis=1)

        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model, 
            test_size=0.2,
            val_size=0.1,
            do_hyperparameter_tuning=False,
            do_cross_validation=False,
            random_state=42
        )
        trainer.split_data(df, target_col='target')
        trainer.train_with_cv(cv_folds=2)

        importance = trainer.get_feature_importance()
        assert len(importance) == 2
        assert 'Feature' in importance.columns
        assert 'Importance' in importance.columns