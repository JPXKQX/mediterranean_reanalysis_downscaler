from typing import List, Tuple

from torch import nn

import logging

logger = logging.getLogger(__name__)


def load_trained_model(class_name: str = None, model_dir: str = None) -> nn.Module:
    """Load a trained model and return it in evaluation mode.

    Args:
    ----
        class_name (str): Name of the model class. Options are
        model_dir (str): Directory where the model is stored.

    Returns:
    -------
        nn.Module: the model in evaluation mode.
    """
    if class_name.lower() == "convswin2sr":
        from deepr.model.conv_swin2sr import ConvSwin2SR

        model = ConvSwin2SR.from_pretrained(model_dir)
    else:
        logger.warning(
            f"The class_name {class_name} is not implemented. "
            f"Options are 'convbaseline', 'convswin2sr' and 'cddpm."
        )
        return None
    model.eval()
    return model


def get_neural_network(
    class_name: str,
    kwargs: dict,
    input_shape: Tuple[int] = None,
    sample_size: Tuple[int] = None,
    out_channels: int = None,
    static_covariables: List[str] = None,
) -> nn.Module:
    """Get neural network.

    Given a class name and a dictionary of keyword arguments, returns an instance of a
    neural network. Current options are: "UNet".

    Arguments
    ---------
    class_name : str
        The name of the neural network class to use.
    kwargs : dict
        Dictionary of keyword arguments to pass to the neural network constructor.
    input_shape : Optional[tuple]
        Sample size of the input samples.
    sample_size : Optional[tuple]
        Sample size of the target samples.
    out_channels : Optional[int]
        Output channels of the target samples.

    Returns
    -------
    model: nn.Module
        An instance of a neural network.

    Raises:
    ------
        NotImplementedError: If the specified neural network class is not implemented.
    """
    if "sample_size" in kwargs:
        kwargs["sample_size"] = tuple(kwargs["sample_size"])
    elif sample_size is None:
        raise ValueError(f"sample_size must be specified for {class_name}")
    else:
        kwargs["sample_size"] = sample_size

    if "out_channels" not in kwargs and out_channels is not None:
        kwargs["out_channels"] = out_channels

    if class_name.lower() == "convswin2sr":
        from deepr.model.conv_swin2sr import ConvSwin2SR, ConvSwin2SRConfig

        kwargs["num_channels"] = kwargs.pop("out_channels")
        if input_shape is not None:
            kwargs["input_shape"] = input_shape

        if static_covariables is not None:
            kwargs["num_high_res_covars"] = len(static_covariables)

        cfg = ConvSwin2SRConfig(**kwargs)
        return ConvSwin2SR(cfg)
    else:
        raise NotImplementedError(f"{class_name} is not implemented")
