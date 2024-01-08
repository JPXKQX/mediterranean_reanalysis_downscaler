# -*- coding: utf-8 -*-
# import project config.py
"""
   Module to prepare the dataset
"""
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import reanalysis_downscaler.config as cfg


def set_scalers(data_config, train_features, train_label):
    scaling_cfg = data_config.features_configuration.get("standardization", {})
    if train_features is not None and scaling_cfg.get("to_do", False):
        scaler_filename = Path(scaling_cfg["cache_folder"])
        os.makedirs(scaler_filename, exist_ok=True)
        scaler_filename = (
            scaler_filename / f"{scaling_cfg['method']}_features_scale.pkl"
        )
        features_scaler = XarrayStandardScaler(
            train_features, scaling_cfg["method"], scaler_filename
        )
    else:
        logger.info("No features standardization.")
        features_scaler = None

    scaling_cfg = data_config.label_configuration.get("standardization", {})
    if train_label is not None and scaling_cfg.get("to_do", False):
        scaler_filename = Path(scaling_cfg["cache_folder"])
        os.makedirs(scaler_filename, exist_ok=True)
        scaler_filename = (
            scaler_filename / f"{scaling_cfg['method']}_label_scale.pkl"
        )
        label_scaler = XarrayStandardScaler(
            train_label, scaling_cfg["method"], scaler_filename,
        )
    elif (
        features_scaler is not None and features_scaler.scaling_method == "domain-wise"
    ):
        logger.info(
            "No label standardization. The labels are scaled with the features scaler."
        )
        label_scaler = copy.deepcopy(features_scaler)
        label_scaler.scaling_files = train_label
    else:
        logger.info("No label standardization.")
        label_scaler = None

    return features_scaler, label_scaler


def main(data_config):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Get features and labels from data_configuration dictionary.")
    train_features, val_features, test_features = data_config.get_features()
    train_label, val_label, test_label = data_config.get_labels()

    # Set scaler with training set
    features_scaler, label_scaler = set_scalers(
        data_config, train_features, train_label
    )

    # Load static variables
    logger.info("Define the DataGenerator object.")
    static_features = data_config.get_static_features()
    static_label = data_config.get_static_label()
    add_aux = {"time": True}

    # Define DataGenerator objects
    data_gen_train = DataGenerator(
        train_features,
        train_label,
        add_aux,
        features_scaler,
        label_scaler,
    )
    data_gen_val = DataGenerator(
        val_features, val_label, add_aux, features_scaler, label_scaler
    )
    data_gen_test = DataGenerator(
        test_features,
        test_label,
        add_aux,
        features_scaler,
        label_scaler,
        shuffle=False,
    )

    return data_gen_train, data_gen_val, data_gen_test


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
