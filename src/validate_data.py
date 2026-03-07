import great_expectations as gx
import pandas as pd
from pathlib import Path
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeOfType
)

def validate_data_func():
    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'raw_data.csv'

    context = gx.get_context()
    df = pd.read_csv(data_path)

    datasource = context.data_sources.add_pandas(name="my_pandas_datasource")
    data_asset = datasource.add_dataframe_asset(name="raw_data_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")

    suite = context.suites.add(gx.ExpectationSuite(name="my_suite"))

    suite.add_expectation(ExpectColumnValuesToBeBetween(column="price", min_value=0))
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="stock", min_value=0, max_value=1000))
    suite.add_expectation(ExpectColumnValuesToBeOfType(column="stock", type_="int64"))
    for col in ["model", "price", "stock"]:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="validation",
            data=batch_definition,
            suite=suite
        )
    )

    checkpoints = context.checkpoints.add(
        gx.Checkpoint(
            name="my_checkpoint",
            validation_definitions=[validation_def],
            result_format="SUMMARY"
        )
    )

    results = checkpoints.run(batch_parameters={"dataframe": df})

    return results.success