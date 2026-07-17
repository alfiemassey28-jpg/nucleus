import pandas as pd

from cdk.instruments.platereader import PlateReaderData, PlateReaderDataFormat
from cdk.instruments.platereader.constants import PlateReaderDataColumn
from cdk.instruments.platereader.lineage import Quantity

WELL = PlateReaderDataColumn.WELL
READ = PlateReaderDataColumn.READ
DATA = PlateReaderDataColumn.DATA
DATA_RATIO = PlateReaderDataColumn.DATA_RATIO


def _endpoint_block():
    df = pd.DataFrame(
        [
            {WELL: "A1", READ: "490,535", DATA: 3000.0},
            {WELL: "A1", READ: "440,535", DATA: 1000.0},
        ]
    )
    return PlateReaderData(
        data=df,
        platemap_columns=set(),
        metadata_columns={READ},
        format=PlateReaderDataFormat.ENDPOINT,
    )


def test_ratio_returns_new_instance_with_lineage():
    block = _endpoint_block()
    out = block.ratio("490,535", "440,535")
    assert out is not block  # immutable, new instance
    assert block.data.shape[0] == 2  # original untouched
    assert out.data.shape[0] == 1  # contracted
    assert out.active_data_column == DATA_RATIO
    assert out.data[DATA_RATIO].iloc[0] == 3.0
    last = out.lineage[-1]
    assert last.kind == "ratio"
    assert last.output_role == Quantity("Ratio")
    assert last.params["numerator_read"] == "490,535"


def test_ratio_default_name():
    out = _endpoint_block().ratio("490,535", "440,535")
    assert out.data[READ].iloc[0] == "490,535 / 440,535"
