"""
Pipeline principal (CLI) pour l'entraînement et la génération de soumissions.
"""

import argparse

from config import TRAIN_TEST_SPLIT_RATIO
from data_loading import load_data, explore_data
from data_preparation import prepare_data, split_temporal_data, prepare_features_for_modeling
from feature_engineering import create_all_features
from model_training import train_base_models, create_results_dataframe
from cross_validation import perform_time_series_cross_validation, get_feature_importance
from predictions import train_final_model, generate_predictions, save_submissions


def run_pipeline(run_explore=True, run_base_models=True, run_cv=True):
    # 1) Load
    X_train, y_train, X_test, sample_submission = load_data()
    if run_explore:
        explore_data(X_train, y_train, X_test)

    # 2) Prepare
    train_data, X_test = prepare_data(X_train, y_train, X_test)

    # 3) Features
    train_data, X_test, all_features = create_all_features(train_data, X_test)

    # 4) Split time-aware
    train_mask, val_mask = split_temporal_data(train_data, TRAIN_TEST_SPLIT_RATIO)

    # 5) Prepare matrices
    X, y, X_test_features = prepare_features_for_modeling(train_data, X_test, all_features)
    X_train_split = X.loc[train_mask]
    y_train_split = y.loc[train_mask]
    X_val_split = X.loc[val_mask]
    y_val_split = y.loc[val_mask]

    # 6) Base models
    if run_base_models:
        results = train_base_models(X_train_split, y_train_split, X_val_split, y_val_split)
        create_results_dataframe(results)

    # 7) CV (LightGBM)
    cv_models = []
    if run_cv:
        cv_scores, cv_models, mean_score, std_score = perform_time_series_cross_validation(
            X, y, train_data, all_features
        )
        get_feature_importance(cv_models, all_features, top_n=30)

    # 8) Final model + predictions
    model_final = train_final_model(X, y, all_features)
    pred_final, pred_ensemble = generate_predictions(model_final, cv_models, X_test_features)

    # 9) Save submissions
    submissions_info = save_submissions(
        {
            'lgbm_final': pred_final,
            'ensemble': pred_ensemble,
        },
        sample_submission,
    )

    return submissions_info


def main():
    parser = argparse.ArgumentParser(description="Main training pipeline")
    parser.add_argument("--no-explore", action="store_true", help="Skip data exploration")
    parser.add_argument("--no-base-models", action="store_true", help="Skip base model training")
    parser.add_argument("--no-cv", action="store_true", help="Skip CV")
    args = parser.parse_args()

    run_pipeline(
        run_explore=not args.no_explore,
        run_base_models=not args.no_base_models,
        run_cv=not args.no_cv,
    )


if __name__ == "__main__":
    main()
