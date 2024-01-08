# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields
from marshmallow import Schema, INCLUDE

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data')
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')

# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select a file for the prediction"
    )

    # to be able to provide an URL for prediction
    urls = fields.Url(
        required=False,
        missing=None,
        description="Provide an URL of the data for the prediction"
    )
    
    # an input parameter for prediction
    arg1 = fields.Str(
        required=False,
        missing="tas",
        description="Variable to output"
    )

# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    num_epochs = fields.Integer(
        required=True,
        missing=10,
        description="Number of epochs for training."
    )
    batch_size = fields.Integer(
        required=True,
        missing=8,
        description="Number of samples per batch."
    )
    num_workers = fields.Integer(
        required=False,
        missing=4,
        description="Number of worker processes for data loading."
    )
    gradient_accumulation_steps = fields.Integer(
        required=False,
        missing=1,
        description="Number of steps for gradient accumulation."
    )
    num_samples = fields.Integer(
        required=False,
        missing=None,  # Replace with an appropriate default value
        description="Total number of samples in the dataset."
    )
    learning_rate = fields.Float(
        required=False,
        missing=0.0005,
        description="Learning rate for the optimizer."
    )
    lr_warmup_steps = fields.Integer(
        required=False,
        missing=1000,
        description="Number of warm-up steps for learning rate."
    )
    mixed_precision = fields.String(
        required=False,
        missing="fp16",
        description="Whether to use mixed precision for training."
    )
