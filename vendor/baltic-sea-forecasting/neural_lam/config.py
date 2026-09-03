# Standard library
import dataclasses
from pathlib import Path
from typing import Dict, List, Union

# Third-party
import dataclass_wizard

# Local
from .datastore import (
    DATASTORES,
    MDPDatastore,
    init_datastore,
)


class DatastoreKindStr(str):
    VALID_KINDS = DATASTORES.keys()

    def __new__(cls, value):
        if value not in cls.VALID_KINDS:
            raise ValueError(f"Invalid datastore kind: {value}")
        return super().__new__(cls, value)


@dataclasses.dataclass
class DatastoreSelection:
    """
    Configuration for selecting a datastore to use with neural-lam.

    Attributes
    ----------
    kind : DatastoreKindStr
        The kind of datastore to use, currently `mdp` is implemented.
    config_path : str
        The path to the configuration file for the selected datastore, this is
        assumed to be relative to the configuration file for neural-lam.
    """

    kind: DatastoreKindStr
    config_path: str


@dataclasses.dataclass
class ManualStateFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    the weights are manually specified.

    Attributes
    ----------
    weights : Dict[str, float]
        Manual weights for the state features.
    """

    weights: Dict[str, float]


@dataclasses.dataclass
class UniformFeatureWeighting:
    """
    Configuration for weighting the state features in the loss function where
    all state features are weighted equally.
    """

    pass


@dataclasses.dataclass
class OutputClamping:
    """
    Configuration for clamping the output of the model.

    Attributes
    ----------
    lower : Dict[str, float]
        The minimum value to clamp each output feature to.
    upper : Dict[str, float]
        The maximum value to clamp each output feature to.
    """

    lower: Dict[str, float] = dataclasses.field(default_factory=dict)
    upper: Dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DensityChannel:
    """
    Density channel for sea ice/wave variables.

    A binary density channel (1 = ice present, 0 = absent) is constructed from
    `reference_var` and appended to the state features. The model predicts the
    density channel alongside all other variables. During autoregressive rollout
    the predicted density is thresholded at 0.5: where density < 0.5, all
    `associated_vars` (and the density channel itself) are set to zero, ensuring
    clean ice-free regions without softplus drift.

    Attributes
    ----------
    reference_var : str
        State variable used to determine ice presence (e.g. "siconc").
        Density = 1 where reference_var > 0 in physical space.
    associated_vars : List[str]
        State variables to zero out where density < 0.5 (e.g. ["siconc",
        "sithick"]).  Should include the reference_var itself.
    """

    reference_var: str = "siconc"
    associated_vars: List[str] = dataclasses.field(
        default_factory=lambda: ["siconc", "sithick"]
    )


@dataclasses.dataclass
class TrainingConfig:
    """
    Configuration related to training neural-lam

    Attributes
    ----------
    state_feature_weighting : Union[ManualStateFeatureWeighting,
                                    UnformFeatureWeighting]
        The method to use for weighting the state features in the loss
        function. Defaults to uniform weighting (`UnformFeatureWeighting`, i.e.
        all features are weighted equally).
    """

    state_feature_weighting: Union[
        ManualStateFeatureWeighting, UniformFeatureWeighting
    ] = dataclasses.field(default_factory=UniformFeatureWeighting)

    output_clamping: OutputClamping = dataclasses.field(
        default_factory=OutputClamping
    )

    density_channel: Union[DensityChannel, None] = None

    # If True, weigh grid-point contributions in loss/metrics by cos(latitude)
    # (normalized to unit mean over the interior grid), approximating equal-area
    # weighting on the sphere. If False (default), all interior grid points are
    # weighted uniformly in the loss.
    lat_weighted_loss: bool = False


@dataclasses.dataclass
class NeuralLAMConfig(dataclass_wizard.JSONWizard, dataclass_wizard.YAMLWizard):
    """
    Dataclass for Neural-LAM configuration. This class is used to load and
    store the configuration for using Neural-LAM.

    Attributes
    ----------
    datastore : DatastoreSelection
        The configuration for the datastore to use.
    datastore_boundary : Union[DatastoreSelection, None]
        The configuration for the boundary datastore to use, if any. If None,
        no boundary datastore is used.
    datastore_atmosphere : Union[DatastoreSelection, None]
        The configuration for the atmosphere datastore to use, if any. If None,
        no atmosphere datastore is used.
    training : TrainingConfig
        The configuration for training the model.
    """

    datastore: DatastoreSelection
    datastore_boundary: Union[DatastoreSelection, None] = None
    datastore_atmosphere: Union[DatastoreSelection, None] = None
    statistics_datastore: Union[DatastoreSelection, None] = None
    statistics_datastore_boundary: Union[DatastoreSelection, None] = None
    statistics_datastore_atmosphere: Union[DatastoreSelection, None] = None
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    class _(dataclass_wizard.JSONWizard.Meta):
        """
        Define the configuration class as a JSON wizard class.

        Together `tag_key` and `auto_assign_tags` enable that when a `Union` of
        types are used for an attribute, the specific type to deserialize to
        can be specified in the serialised data using the `tag_key` value. In
        our case we call the tag key `__config_class__` to indicate to the
        user that they should pick a dataclass describing configuration in
        neural-lam. This Union-based selection allows us to support different
        configuration attributes for different choices of methods for example
        and is used when picking between different feature weighting methods in
        the `TrainingConfig` class. `auto_assign_tags` is set to True to
        automatically set that tag key (i.e. `__config_class__` in the config
        file) should just be the class name of the dataclass to deserialize to.
        """

        tag_key = "__config_class__"
        auto_assign_tags = True
        # ensure that all parts of the loaded configuration match the
        # dataclasses used
        # TODO: this should be enabled once
        # https://github.com/rnag/dataclass-wizard/issues/137 is fixed, but
        # currently cannot be used together with `auto_assign_tags` due to a
        # bug it seems
        # raise_on_unknown_json_key = True


class InvalidConfigError(Exception):
    pass


def load_config_and_datastores(
    config_path: str,
) -> tuple[NeuralLAMConfig, MDPDatastore]:
    """
    Load the neural-lam configuration and the datastores specified in the
    configuration.

    Parameters
    ----------
    config_path : str
        Path to the Neural-LAM configuration file.

    Returns
    -------
    tuple[NeuralLAMConfig, MDPDatastore]
        The Neural-LAM configuration and the loaded datastores.
    """
    try:
        config = NeuralLAMConfig.from_yaml_file(config_path)
    except dataclass_wizard.errors.UnknownJSONKey as ex:
        raise InvalidConfigError(
            "There was an error loading the configuration file at "
            f"{config_path}. "
        ) from ex
    # datastore config is assumed to be relative to the config file
    datastore_config_path = (
        Path(config_path).parent / config.datastore.config_path
    )
    datastore = init_datastore(
        datastore_kind=config.datastore.kind, config_path=datastore_config_path
    )

    if config.datastore_boundary is not None:
        datastore_boundary_config_path = (
            Path(config_path).parent / config.datastore_boundary.config_path
        )
        datastore_boundary = init_datastore(
            datastore_kind=config.datastore_boundary.kind,
            config_path=datastore_boundary_config_path,
        )
    else:
        datastore_boundary = None

    if config.datastore_atmosphere is not None:
        datastore_atmosphere_config_path = (
            Path(config_path).parent / config.datastore_atmosphere.config_path
        )
        datastore_atmosphere = init_datastore(
            datastore_kind=config.datastore_atmosphere.kind,
            config_path=datastore_atmosphere_config_path,
        )
    else:
        datastore_atmosphere = None

    # Optional separate datastores for loading standardization statistics
    # (e.g. training-data stats used to normalize forecast-mode data).
    # When not specified, statistics are loaded from the main datastores.
    if config.statistics_datastore is not None:
        stats_ds_config_path = (
            Path(config_path).parent / config.statistics_datastore.config_path
        )
        statistics_datastore = init_datastore(
            datastore_kind=config.statistics_datastore.kind,
            config_path=stats_ds_config_path,
        )
    else:
        statistics_datastore = None

    if config.statistics_datastore_boundary is not None:
        stats_boundary_config_path = (
            Path(config_path).parent
            / config.statistics_datastore_boundary.config_path
        )
        statistics_datastore_boundary = init_datastore(
            datastore_kind=config.statistics_datastore_boundary.kind,
            config_path=stats_boundary_config_path,
        )
    else:
        statistics_datastore_boundary = None

    if config.statistics_datastore_atmosphere is not None:
        stats_atmosphere_config_path = (
            Path(config_path).parent
            / config.statistics_datastore_atmosphere.config_path
        )
        statistics_datastore_atmosphere = init_datastore(
            datastore_kind=config.statistics_datastore_atmosphere.kind,
            config_path=stats_atmosphere_config_path,
        )
    else:
        statistics_datastore_atmosphere = None

    return (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    )
