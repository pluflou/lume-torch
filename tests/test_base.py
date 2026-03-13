import os
import pytest
import yaml
import torch

from lume_torch.base import LUMETorch
from lume_torch.variables import TorchScalarVariable, TorchNDVariable


class ExampleModel(LUMETorch):
    def _evaluate(self, input_dict):
        pass


class TestBaseModel:
    def test_init(self, simple_variables):
        # init with no variable specification
        with pytest.raises(TypeError):
            _ = LUMETorch()

        # init child class with no _evaluate function
        class NoEvaluateModel(LUMETorch):
            def predict(self, input_dict):
                pass

        with pytest.raises(TypeError):
            _ = NoEvaluateModel(**simple_variables)

        # TODO: move to test_torch_model.py
        # # init child class with input variables missing default value
        # simple_variables_no_default = copy.deepcopy(simple_variables)
        # simple_variables_no_default["input_variables"][0].default_value = None
        # with pytest.raises(ValueError):
        #     _ = ExampleModel(**simple_variables_no_default)

        # init child class with evaluate function
        example_model = ExampleModel(**simple_variables)
        assert example_model.input_variables == simple_variables["input_variables"]
        assert example_model.output_variables == simple_variables["output_variables"]

        # input and output variables sharing names is fine
        input_variables = simple_variables["input_variables"]
        output_variables = simple_variables["output_variables"]
        original_name = input_variables[0].name
        input_variables[0].name = output_variables[0].name
        _ = ExampleModel(**simple_variables)
        input_variables[0].name = original_name

    def test_dict(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        dict_output = example_model.model_dump()
        assert isinstance(dict_output["input_variables"], list)
        assert isinstance(dict_output["output_variables"], list)
        assert len(dict_output["input_variables"]) == 2

    def test_json(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        _ = example_model.json()

    def test_yaml_serialization(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        yaml_output = example_model.yaml()
        dict_output = yaml.safe_load(yaml_output)
        dict_output["input_variables"]["input1"]["variable_class"] = (
            TorchScalarVariable.__name__
        )

        # test loading from yaml
        loaded_model = ExampleModel(**dict_output)
        assert loaded_model == example_model

    def test_file_serialization(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        file = "test_model.yml"
        example_model.dump(file)

        os.remove(file)

    def test_deserialization_from_config(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        file = "test_model.yml"
        _ = example_model.dump(file)
        loaded_model = ExampleModel(file)
        os.remove(file)
        assert loaded_model.input_variables == example_model.input_variables
        assert loaded_model.output_variables == example_model.output_variables

    def test_input_names(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        for i, var in enumerate(simple_variables["input_variables"]):
            assert example_model.input_names.index(var.name) == i

    def test_output_names(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        for i, var in enumerate(simple_variables["output_variables"]):
            assert example_model.output_names.index(var.name) == i

    def test_input_validation(self, simple_variables, monkeypatch):
        example_model = ExampleModel(**simple_variables)
        input_variables = simple_variables["input_variables"]
        input_dict = {input_variables[0].name: 2.0, input_variables[1].name: 1.5}
        example_model.input_validation(input_dict)
        with pytest.raises(TypeError):
            input_dict[input_variables[0].name] = True
            example_model.input_validation(input_dict)

        # setting strictness flag
        assert input_variables[0].default_validation_config == "none"
        with pytest.raises(ValueError):
            # has to be a ConfigEnum type
            example_model.input_validation_config = {input_variables[0].name: "test"}

        # range check with strictness flag
        example_model.input_validation_config = {input_variables[0].name: "error"}
        with pytest.raises(ValueError):
            input_dict[input_variables[0].name] = 6.0
            example_model.input_validation(input_dict)

        # a warning is printed
        example_model.input_validation_config = {input_variables[0].name: "warn"}
        input_dict[input_variables[0].name] = 6.0
        with pytest.warns(UserWarning):
            example_model.input_validation(input_dict)

        # nothing is printed/raised
        example_model.input_validation_config = {input_variables[0].name: "none"}
        input_dict[input_variables[0].name] = 6.0
        example_model.input_validation(input_dict)

    def test_output_validation(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        output_variables = simple_variables["output_variables"]
        output_dict = {output_variables[0].name: 3.0, output_variables[1].name: 1.7}
        example_model.output_validation(output_dict)
        with pytest.raises(TypeError):
            output_dict[output_variables[0].name] = "test"
            example_model.output_validation(output_dict)

    def test_output_validation_unknown_name_raises(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        output_variables = simple_variables["output_variables"]
        with pytest.raises(ValueError, match="not found"):
            example_model.output_validation(
                {output_variables[0].name: 1.0, "nonexistent_output": 2.0}
            )


class TestValidateDictPerVariable:
    """Tests for LUMETorch._validate_dict_per_variable — the shared unbatching helper."""

    @pytest.fixture
    def scalar_model(self, simple_variables):
        return ExampleModel(**simple_variables)

    @pytest.fixture
    def nd_model(self):
        input_vars = [TorchNDVariable(name="image", shape=(3, 4))]
        output_vars = [TorchNDVariable(name="feat", shape=(2, 2))]
        return ExampleModel(input_variables=input_vars, output_variables=output_vars)

    @pytest.fixture
    def readonly_model(self):
        input_vars = [
            TorchScalarVariable(name="fixed", default_value=1.0, read_only=True),
            TorchScalarVariable(name="free", default_value=2.0),
        ]
        output_vars = [TorchScalarVariable(name="out")]
        return ExampleModel(input_variables=input_vars, output_variables=output_vars)

    def test_unknown_variable_name_raises(self, scalar_model, simple_variables):
        input_variables = simple_variables["input_variables"]
        with pytest.raises(ValueError, match="not found"):
            scalar_model._validate_dict_per_variable(
                {input_variables[0].name: 1.0, "does_not_exist": 2.0},
                scalar_model.input_variables,
                None,
            )

    def test_unbatched_scalar_passes(self, scalar_model, simple_variables):
        input_variables = simple_variables["input_variables"]
        scalar_model._validate_dict_per_variable(
            {input_variables[0].name: torch.tensor(1.0)},
            scalar_model.input_variables,
            None,
        )

    def test_batched_scalar_tensor_validates(self, scalar_model, simple_variables):
        input_variables = simple_variables["input_variables"]
        # shape (N,) — batched scalar
        scalar_model._validate_dict_per_variable(
            {input_variables[0].name: torch.tensor([1.0, 2.0, 3.0])},
            scalar_model.input_variables,
            None,
        )

    def test_batched_scalar_wrong_type_raises(self, scalar_model, simple_variables):
        input_variables = simple_variables["input_variables"]
        # boolean tensor — invalid dtype for TorchScalarVariable (raises ValueError for dtype, TypeError for type)
        with pytest.raises((TypeError, ValueError)):
            scalar_model._validate_dict_per_variable(
                {input_variables[0].name: torch.tensor([True, False])},
                scalar_model.input_variables,
                None,
            )

    def test_unbatched_nd_tensor_validates(self, nd_model):
        nd_model._validate_dict_per_variable(
            {"image": torch.zeros(3, 4)},
            nd_model.input_variables,
            None,
        )

    def test_batched_nd_tensor_validates(self, nd_model):
        # shape (N, H, W) — batched ND tensor
        nd_model._validate_dict_per_variable(
            {"image": torch.zeros(5, 3, 4)},
            nd_model.input_variables,
            None,
        )

    def test_batched_nd_wrong_shape_raises(self, nd_model):
        # Unbatched tensor with wrong shape (ndim == len(var.shape)) goes through
        # validate_value directly and raises ValueError for shape mismatch.
        with pytest.raises(ValueError, match="Expected shape"):
            nd_model._validate_dict_per_variable(
                {"image": torch.zeros(3, 99)},  # ndim=2 == len(shape), not batched
                nd_model.input_variables,
                None,
            )

    def test_batched_nd_wrong_dtype_raises(self, nd_model):
        with pytest.raises(ValueError, match="Expected dtype"):
            nd_model._validate_dict_per_variable(
                {"image": torch.zeros(5, 3, 4, dtype=torch.float64)},
                nd_model.input_variables,
                None,
            )

    def test_read_only_batched_all_match(self, readonly_model):
        # All batch elements equal the default — should pass
        readonly_model._validate_dict_per_variable(
            {
                "fixed": torch.tensor([1.0, 1.0, 1.0]),
                "free": torch.tensor([2.0, 3.0, 4.0]),
            },
            readonly_model.input_variables,
            None,
        )

    def test_read_only_batched_mismatch_raises(self, readonly_model):
        # One batch element differs from the default — should raise
        with pytest.raises(ValueError, match="read-only"):
            readonly_model._validate_dict_per_variable(
                {"fixed": torch.tensor([1.0, 99.0]), "free": torch.tensor([2.0, 2.0])},
                readonly_model.input_variables,
                None,
            )

    def test_missing_variable_in_dict_is_skipped(self, scalar_model, simple_variables):
        # Variables not present in data_dict are silently skipped
        input_variables = simple_variables["input_variables"]
        scalar_model._validate_dict_per_variable(
            {input_variables[0].name: torch.tensor(1.0)},  # only one of two vars
            scalar_model.input_variables,
            None,
        )
