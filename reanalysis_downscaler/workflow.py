import copy
import json
import os
from pathlib import Path
from typing import Tuple
import logging
from torch import nn
from torch.utils.data import DataLoader

from reanalysis_downscaler.dataset.configuration import DataConfiguration
from reanalysis_downscaler.data.generator import DataGenerator
from reanalysis_downscaler.dataset.scaler import XarrayStandardScaler
from reanalysis_downscaler.models.configs import TrainingConfig
from reanalysis_downscaler.model.nn_trainer import train_nn
from reanalysis_downscaler.utilities.yml import read_yaml_file


logger = logging.getLogger(__name__)


class MainPipeline:
    def __init__(self, configuration_file: Path):
        """
        Initialize the MainPipeline class.

        Parameters
        ----------
        configuration_file : Path
            Path to the configuration file.
        """
        logger.info(f"Reading experiment configuration from file {configuration_file}.")
        configuration = read_yaml_file(configuration_file)
        self.data_config = DataConfiguration(configuration["data_configuration"])
        train_config = configuration.get("training_configuration", {})
        self.pipeline_type = train_config.get("type", None)
        self.model_config = train_config.get("model_configuration", None)
        if "training_parameters" in train_config.keys():
            self.train_config = TrainingConfig(
                **train_config.get("training_parameters", {})
            )
        else:
            self.train_config = None
        self.validation_config = configuration.get("validation_configuration", {})
        self.inference_config = configuration.get("inference_configuration", {})
        self.features_scaler = None
        self.label_scaler = None

    def set_scalers(self, train_features, train_label):
        scaling_cfg = self.data_config.features_configuration.get("standardization", {})
        if train_features is not None and scaling_cfg.get("to_do", False):
            scaler_filename = Path(scaling_cfg["cache_folder"])
            os.makedirs(scaler_filename, exist_ok=True)
            scaler_filename = (
                scaler_filename / f"{scaling_cfg['method']}_features_scale.pkl"
            )
            self.features_scaler = XarrayStandardScaler(
                train_features, scaling_cfg["method"], scaler_filename
            )
        else:
            logger.info("No features standardization.")
            self.features_scaler = None

        scaling_cfg = self.data_config.label_configuration.get("standardization", {})
        if train_label is not None and scaling_cfg.get("to_do", False):
            scaler_filename = Path(scaling_cfg["cache_folder"])
            os.makedirs(scaler_filename, exist_ok=True)
            scaler_filename = (
                scaler_filename / f"{scaling_cfg['method']}_label_scale.pkl"
            )
            self.label_scaler = XarrayStandardScaler(
                train_label,
                scaling_cfg["method"],
                scaler_filename,
            )
        elif (
            self.features_scaler is not None
            and self.features_scaler.scaling_method == "domain-wise"
        ):
            logger.info(
                "No label standardization. The labels are scaled with the features scaler."
            )
            self.label_scaler = copy.deepcopy(self.features_scaler)
            self.label_scaler.scaling_files = train_label
        else:
            logger.info("No label standardization.")
            self.label_scaler = None

        if self.train_config is not None and self.train_config.output_dir is not None:
            output_dir = Path(self.train_config.output_dir)
            scales = {
                "features": self.features_scaler.to_dict(),
                "label": self.label_scaler.to_dict(),
            }
            with open(output_dir / "training_scale.json", "w") as f:
                json.dumps(scales, f, indent=4)

    def get_dataset(self) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
        """
        Initialize the data_loader for the pipeline.

        Returns
        -------
        data_generator : Tuple[DataGenerator, DataGenerator, DataGenerator]
            The initialized DataGenerator objects for training, validation, and testing.
        """
        logger.info("Get features and labels from data_configuration dictionary.")
        train_features, val_features, test_features = self.data_config.get_features()
        train_label, val_label, test_label = self.data_config.get_labels()

        # Set scaler with training set
        self.set_scalers(train_features, train_label)

        # Load static variables
        logger.info("Define the DataGenerator object.")
        static_features = self.data_config.get_static_features()
        static_label = self.data_config.get_static_label()
        add_aux = self.define_aux_data(self.data_config, static_features, static_label)

        # Define DataGenerator objects
        data_gen_train = DataGenerator(
            train_features,
            train_label,
            add_aux,
            self.features_scaler,
            self.label_scaler,
        )
        data_gen_val = DataGenerator(
            val_features, val_label, add_aux, self.features_scaler, self.label_scaler
        )
        data_gen_test = DataGenerator(
            test_features,
            test_label,
            add_aux,
            self.features_scaler,
            self.label_scaler,
            shuffle=False,
        )
        return data_gen_train, data_gen_val, data_gen_test

    def train_model(
        self,
        dataset_train: DataGenerator,
        dataset_val: DataGenerator,
    ):
        """
        Train a Super Resolution (SR) model with the given dataset.

        Parameters
        ----------
        dataset_train : DataGenerator
            The dataset to train the model on.
        dataset_val: DataGenerator
            The dataset to validate the model on.

        Returns
        -------
        model : nn.Module | SuperResolutionDenoiseDiffusion
            The trained neural network.
        """
        model_cfg = self.model_config.pop("neural_network")

        model = get_neural_network(
            **model_cfg,
            out_channels=dataset_train.output_channels,
            sample_size=dataset_train.output_shape,
            input_shape=dataset_train.input_shape,
            static_covariables=self.train_config.static_covariables,
        )

        return train_nn(
            self.train_config,
            model,
            dataset_train,
            dataset_val,
            self.data_config.get_plain_dict(),
        )

    def run_pipeline(self):
        """Run the pipeline."""
        dataset_train, dataset_val, dataset_test = self.get_dataset()
        model, repo_name = self.train_model(dataset_train, dataset_val)
        self.validate_model(
            model, dataset_test, self.validation_config, hf_repo_name=repo_name
        )

