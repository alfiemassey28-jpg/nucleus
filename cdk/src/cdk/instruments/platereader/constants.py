from enum import StrEnum
from ..platemap import PlatemapColumn

class MetadataColumn(StrEnum):
    READER_TYPE = "Reader Type"
    READER_ID = "Reader ID"
    PLATE_TYPE = "Plate Type"
    PLATE_ID = "Plate ID"
    DATE = "Start Time"


class PlateReaderDataColumn(StrEnum):
    # Read identity
    READ_NAME = "Read Name"
    READ_MODALITY = "Read Modality"
    # Standard read optics
    GAIN = "Gain"
    EXCITATION = "Excitation Wavelength (nm)"
    EXCITATION_BW = "Excitation Bandwidth (nm)"
    EXCITATION_OPTICS = "Excitation Optics"
    EMISSION = "Emission Wavelength (nm)"
    EMISSION_BW = "Emission Bandwidth (nm)"
    EMISSION_OPTICS = "Emission Optics"
    GEOMETRY = "Read Geometry"
    READ_HEIGHT = "Read Height (mm)"
    ABSORBANCE_WV = "Absorbance Wavelength (nm)"
    # Data
    WELL = "Well"
    DATA = "Data"
    DATA_BLANKED = "Data_blanked"
    DATA_NORMALIZED = "Data_normalized"
    DATA_RATIO = "Data_ratio"
    READ = "Read"
    TIME = "Time"
    TEMPERATURE = "Temperature"
    #
    NORMALIZED_TO = "Normalized To"


class ReadModality(StrEnum):
    FLUORESCENCE = "Fluorescence"
    ABSORBANCE = "Absorbance"
    LUMINESCENCE = "Luminescence"


DEFAULT_Y_LABELS: dict = {
    ReadModality.FLUORESCENCE: ("Fluorescence", "RFU"),
    ReadModality.ABSORBANCE: ("Absorbance", "AU"),
    ReadModality.LUMINESCENCE: ("Luminescence", "AU"),
}


REQUIRED_DATA_COLUMNS: frozenset = frozenset(
    {
        PlateReaderDataColumn.WELL,
        PlateReaderDataColumn.DATA,
        PlateReaderDataColumn.READ,
    }
)
REQUIRED_METADATA_COLUMNS: frozenset = frozenset(MetadataColumn)
REQUIRED_READ_COLUMNS: frozenset = frozenset(
    {
        PlateReaderDataColumn.READ_NAME,
        PlateReaderDataColumn.READ_MODALITY,
    }
)

STANDARD_READ_COLUMNS_BY_MODALITY: dict[ReadModality, frozenset] = {
    ReadModality.FLUORESCENCE: frozenset(
        {
            PlateReaderDataColumn.GAIN,
            PlateReaderDataColumn.EXCITATION,
            PlateReaderDataColumn.EXCITATION_BW,
            PlateReaderDataColumn.EXCITATION_OPTICS,
            PlateReaderDataColumn.EMISSION,
            PlateReaderDataColumn.EMISSION_BW,
            PlateReaderDataColumn.EMISSION_OPTICS,
            PlateReaderDataColumn.GEOMETRY,
            PlateReaderDataColumn.READ_HEIGHT,
        }
    ),
    ReadModality.LUMINESCENCE: frozenset(
        {
            PlateReaderDataColumn.GAIN,
            PlateReaderDataColumn.EMISSION,
            PlateReaderDataColumn.EMISSION_BW,
            PlateReaderDataColumn.GEOMETRY,
            PlateReaderDataColumn.READ_HEIGHT,
        }
    ),
    ReadModality.ABSORBANCE: frozenset(
        {
            PlateReaderDataColumn.ABSORBANCE_WV,
            PlateReaderDataColumn.READ_HEIGHT,
        }
    ),
}
STANDARD_READ_COLUMNS: frozenset = frozenset().union(
    *STANDARD_READ_COLUMNS_BY_MODALITY.values()
)


class KineticOutput(StrEnum):
    """Canonical flat column names for kinetic-fit outputs (single source of truth)."""

    STEADY_STATE = "Steady State"  # plateau level
    COMPLETION_TIME = "Completion Time"  # time to reach steady state
    COMPLETION_THRESHOLD = "Completion Threshold"  # fraction of plateau used
    MAX_VELOCITY = "Max Velocity"  # max rate
    MAX_VELOCITY_TIME = "Max Velocity Time"
    LAG_TIME = "Lag Time"
    FIT_FUNCTION = "Fit Function"
    FIT_PARAMS = "Fit Params"
    R_SQUARED = "R^2"
    DRIFT = "Drift"
    OFFSET = "Offset"
    GOOD_FIT = "Good Fit"


DEFAULT_COMPLETION_THRESHOLD = 0.95
GOOD_FIT_R_SQUARED = 0.95  # R^2 quality gate for Fit.good_fit (independent of completion threshold)


# Display order: required data → required metadata → required read → standard read
COLUMN_DISPLAY_ORDER: list = [
    PlatemapColumn.DATE,
    PlatemapColumn.EXPERIMENT,
    PlateReaderDataColumn.WELL,
    PlatemapColumn.NAME,
    PlatemapColumn.TYPE,
    PlateReaderDataColumn.TIME,
    PlateReaderDataColumn.DATA,
    PlateReaderDataColumn.READ,
    PlateReaderDataColumn.READ_NAME,
    PlateReaderDataColumn.TEMPERATURE,
    MetadataColumn.READER_TYPE,
    MetadataColumn.READER_ID,
    MetadataColumn.PLATE_TYPE,
    MetadataColumn.PLATE_ID,
    MetadataColumn.DATE,
    KineticOutput.MAX_VELOCITY,
    KineticOutput.MAX_VELOCITY_TIME,
    KineticOutput.LAG_TIME,
    KineticOutput.STEADY_STATE,
    KineticOutput.COMPLETION_TIME,
    KineticOutput.COMPLETION_THRESHOLD,
    KineticOutput.DRIFT,
    KineticOutput.OFFSET,
    KineticOutput.FIT_FUNCTION,
    KineticOutput.FIT_PARAMS,
    KineticOutput.R_SQUARED,
    KineticOutput.GOOD_FIT,
    PlateReaderDataColumn.READ_MODALITY,
    PlateReaderDataColumn.GAIN,
    PlateReaderDataColumn.EXCITATION,
    PlateReaderDataColumn.EXCITATION_BW,
    PlateReaderDataColumn.EXCITATION_OPTICS,
    PlateReaderDataColumn.EMISSION,
    PlateReaderDataColumn.EMISSION_BW,
    PlateReaderDataColumn.EMISSION_OPTICS,
    PlateReaderDataColumn.GEOMETRY,
    PlateReaderDataColumn.READ_HEIGHT,
    PlateReaderDataColumn.ABSORBANCE_WV,
]


def reorder_columns(df):
    """Return df with COLUMN_DISPLAY_ORDER columns first, remainder appended."""
    ordered = [c for c in COLUMN_DISPLAY_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in set(COLUMN_DISPLAY_ORDER)]
    return df[ordered + rest]
