"""Test suite for lume_torch.variables module."""

import json
import pytest
import numpy as np
import torch
from pydantic import ValidationError
from torch.distributions import Normal, Uniform

from lume_torch.variables import (
    ScalarVariable,
    TorchScalarVariable,
    TorchNDVariable,
    DistributionVariable,
    Variable,
    ConfigEnum,
    get_variable,
)


class TestGetVariable:
    """Tests for the get_variable function."""

    def test_get_scalar_variable(self):
        """Test getting ScalarVariable by name."""
        var_cls = get_variable("ScalarVariable")
        assert var_cls is ScalarVariable

    def test_get_torch_scalar_variable(self):
        """Test getting TorchScalarVariable by name."""
        var_cls = get_variable("TorchScalarVariable")
        assert var_cls is TorchScalarVariable

    def test_get_distribution_variable(self):
        """Test getting DistributionVariable by name."""
        var_cls = get_variable("DistributionVariable")
        assert var_cls is DistributionVariable

    def test_get_torch_nd_variable(self):
        """Test getting TorchNDVariable by name."""
        var_cls = get_variable("TorchNDVariable")
        assert var_cls is TorchNDVariable

    def test_get_variable_unknown_name_raises(self):
        """Test that unknown variable name raises KeyError."""
        with pytest.raises(KeyError, match="No variable named"):
            get_variable("UnknownVariable")

    def test_get_variable_creates_instance(self):
        """Test that get_variable returns a class that can be instantiated."""
        # Using ScalarVariable will trigger a deprecation warning
        with pytest.warns(DeprecationWarning, match="ScalarVariable is deprecated"):
            var = get_variable(ScalarVariable.__name__)(name="test")
        assert isinstance(var, ScalarVariable)


class TestScalarVariableAlias:
    """Tests to ensure ScalarVariable is a deprecated subclass of TorchScalarVariable."""

    def test_scalar_variable_is_torch_scalar_variable(self):
        """ScalarVariable should be a deprecated subclass of TorchScalarVariable."""
        # ScalarVariable is now a subclass, not an alias
        assert issubclass(ScalarVariable, TorchScalarVariable)
        with pytest.warns(DeprecationWarning, match="ScalarVariable is deprecated"):
            var = ScalarVariable(name="test_var", default_value=1.0)
        assert isinstance(var, TorchScalarVariable)


class TestTorchScalarVariable:
    """Tests for TorchScalarVariable class."""

    def test_basic_creation(self):
        """Test basic variable creation with minimal parameters."""
        var = TorchScalarVariable(name="test_var")
        assert var.name == "test_var"
        assert var.default_value is None
        assert var.value_range is None
        assert var.read_only is False

    def test_creation_with_all_attributes(self):
        """Test variable creation with all attributes."""
        var = TorchScalarVariable(
            name="full_var",
            default_value=torch.tensor(1.5),
            value_range=(0.0, 10.0),
            unit="meters",
            read_only=True,
            dtype=torch.float32,
        )
        assert var.name == "full_var"
        assert torch.isclose(var.default_value, torch.tensor(1.5))
        assert var.value_range == (0.0, 10.0)
        assert var.unit == "meters"
        assert var.read_only is True
        assert var.dtype == torch.float32

    def test_creation_with_mismatched_dtype_raises(self):
        """Test variable creation with all attributes."""
        with pytest.raises(ValidationError):
            TorchScalarVariable(
                name="full_var",
                default_value=torch.tensor(1.5),
                value_range=(0.0, 10.0),
                unit="meters",
                read_only=True,
                dtype=torch.float64,
            )

    def test_missing_name_raises_validation_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchScalarVariable(default_value=0.1, value_range=(0, 1))

    def test_default_value_as_float(self):
        """Test that float default values are accepted."""
        var = TorchScalarVariable(name="float_var", default_value=2.5)
        assert var.default_value == 2.5

    def test_default_value_as_tensor(self):
        """Test that tensor default values are accepted."""
        var = TorchScalarVariable(name="tensor_var", default_value=torch.tensor(3.0))
        assert torch.isclose(var.default_value, torch.tensor(3.0))

    def test_default_value_out_of_range_raises(self):
        """Test that default value out of range raises ValueError."""
        with pytest.raises(ValueError, match="out of valid range"):
            TorchScalarVariable(
                name="bad_var", default_value=15.0, value_range=(0.0, 10.0)
            )

    # Dtype validation tests
    def test_dtype_float32(self):
        """Test dtype with torch.float32."""
        var = TorchScalarVariable(name="test", dtype=torch.float32)
        assert var.dtype == torch.float32

    def test_dtype_float64(self):
        """Test dtype with torch.float64."""
        var = TorchScalarVariable(name="test", dtype=torch.float64)
        assert var.dtype == torch.float64

    def test_dtype_double(self):
        """Test dtype with torch.double."""
        var = TorchScalarVariable(name="test", dtype=torch.double)
        assert var.dtype == torch.double

    def test_dtype_invalid_type_raises(self):
        """Test that invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be a"):
            TorchScalarVariable(name="test", dtype="invalid_dtype")

    def test_dtype_non_floating_raises(self):
        """Test that non-floating dtype raises ValueError."""
        with pytest.raises(ValueError, match="must be a floating-point type"):
            TorchScalarVariable(name="test", dtype=torch.int32)

    # Value validation tests
    def test_validate_value_float(self):
        """Test validation of float values."""
        var = TorchScalarVariable(name="test")
        var.validate_value(5.0)

    def test_validate_value_int(self):
        """Test validation of int values (allowed as scalars)."""
        var = TorchScalarVariable(name="test")
        var.validate_value(5)

    def test_validate_value_tensor_0d(self):
        """Test validation of 0D tensor."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor(5.0))

    def test_validate_value_tensor_1d_raises(self):
        """Test that 1D (batched) tensor is rejected at the variable level."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="Expected a 0D scalar tensor"):
            var.validate_value(torch.tensor([[5.0], [6.0], [7.0]]))

    def test_validate_value_tensor_batched_raises(self):
        """Test that batched tensor with last dim = 1 is rejected at the variable level."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="Expected a 0D scalar tensor"):
            var.validate_value(torch.tensor([[5.0], [6.0]]))

    def test_validate_value_tensor_invalid_shape_raises(self):
        """Test that tensor with invalid shape raises ValueError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="Expected a 0D scalar tensor"):
            var.validate_value(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    def test_validate_value_invalid_type_raises(self):
        """Test that invalid value type raises TypeError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(TypeError):
            var.validate_value("not_a_number")

    def test_validate_value_non_float_tensor_raises(self):
        """Test that non-float 0D tensor raises ValueError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="floating-point type"):
            var.validate_value(torch.tensor(1))  # int64 0D tensor

    def test_validate_value_wrong_dtype_raises(self):
        """Test that tensor with wrong dtype raises ValueError."""
        var = TorchScalarVariable(name="test", dtype=torch.float64)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(torch.tensor(5.0, dtype=torch.float32))

    # Value range validation tests
    def test_validate_value_in_range(self):
        """Test validation of value within range."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="error"
        )
        var.validate_value(5.0)

    def test_validate_value_out_of_range_error(self):
        """Test that out-of-range value raises error with error config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="error"
        )
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0)

    def test_validate_value_out_of_range_warn(self):
        """Test that out-of-range value warns with warn config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="warn"
        )
        with pytest.warns(UserWarning):
            var.validate_value(15.0)

    def test_validate_value_out_of_range_none(self):
        """Test that out-of-range value passes with none config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="none"
        )
        var.validate_value(15.0)

    def test_validate_value_with_config_override(self):
        """Test that config parameter overrides default_validation_config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="none"
        )
        # Default is "none" but we override with "error"
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config="error")

    def test_validate_value_config_enum_object(self):
        """Test validation with ConfigEnum object instead of string."""
        var = TorchScalarVariable(
            name="test",
            value_range=(0.0, 10.0),
            default_validation_config=ConfigEnum.NULL,
        )
        var.validate_value(15.0)

    def test_value_range_validation(self):
        """Test that value_range min must be <= max."""
        with pytest.raises(ValueError, match="Minimum value"):
            TorchScalarVariable(name="test", value_range=(10.0, 0.0))

    # Read-only validation tests
    # Note: read-only validation is handled by _validate_read_only, not
    # validate_value. validate_value only checks type/dtype/range.
    # The model layer is responsible for calling _validate_read_only separately.
    def test_read_only_matching_value(self):
        """Test read-only variable with matching value."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        var._validate_read_only(5.0)

    def test_read_only_matching_tensor(self):
        """Test read-only variable with matching tensor value."""
        var = TorchScalarVariable(
            name="test", default_value=torch.tensor(5.0), read_only=True
        )
        var._validate_read_only(torch.tensor(5.0))

    def test_read_only_non_matching_value_raises(self):
        """Test read-only variable with non-matching value raises."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        with pytest.raises(ValueError, match="read-only"):
            var._validate_read_only(10.0)

    def test_read_only_no_default_raises(self):
        """Test read-only variable without default raises."""
        var = TorchScalarVariable(name="test", read_only=True)
        with pytest.raises(ValueError, match="no default value"):
            var._validate_read_only(5.0)

    def test_read_only_with_tensor_default_float_value(self):
        """Test read-only with tensor default but float value."""
        var = TorchScalarVariable(
            name="test", default_value=torch.tensor(5.0), read_only=True
        )
        var._validate_read_only(5.0)

    def test_read_only_with_float_default_tensor_value(self):
        """Test read-only with float default but tensor value."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        var._validate_read_only(torch.tensor(5.0))

    def test_read_only_tensor_near_default_within_tolerance(self):
        """Test read-only with values very close to default within tolerance."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        # Value very close to default (within 1e-9 tolerance)
        var._validate_read_only(torch.tensor(5.0 + 1e-10))

    def test_model_dump(self):
        """Test model_dump includes variable_class."""
        var = TorchScalarVariable(name="test", default_value=1.0)
        dump = var.model_dump()
        assert "variable_class" in dump
        assert dump["variable_class"] == "TorchScalarVariable"
        assert dump["name"] == "test"

    def test_dtype_none_allows_any_float_dtype(self):
        """Test that dtype=None allows any floating-point dtype."""
        var = TorchScalarVariable(name="test", dtype=None)
        var.validate_value(torch.tensor(5.0, dtype=torch.float32))
        var.validate_value(torch.tensor(5.0, dtype=torch.float64))

    def test_model_dump_includes_all_attributes(self):
        """Test model_dump includes all relevant attributes."""
        var = TorchScalarVariable(
            name="test",
            default_value=5.0,
            value_range=(0.0, 10.0),
            unit="meters",
            read_only=False,
        )
        dump = var.model_dump()
        assert dump["name"] == "test"
        assert dump["default_value"] == 5.0
        assert dump["value_range"] == (0.0, 10.0)
        assert dump["unit"] == "meters"
        assert dump["read_only"] is False

    def test_numpy_float_value(self):
        """Test validation of numpy float values."""
        var = TorchScalarVariable(name="test")
        var.validate_value(np.float64(5.0))


class TestTorchNDVariable:
    """Tests for TorchNDVariable class."""

    def test_basic_creation(self):
        """Test basic ND variable creation."""
        var = TorchNDVariable(name="test_nd", shape=(10, 20))
        assert var.name == "test_nd"
        assert var.shape == (10, 20)
        assert var.dtype == torch.float32

    def test_missing_name_raises_validation_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchNDVariable(shape=(10, 20))

    def test_missing_shape_raises_validation_error(self):
        """Test that missing shape raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchNDVariable(name="test")

    def test_creation_with_default_value(self):
        """Test ND variable creation with default value."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(name="test_nd", shape=(10, 20), default_value=default)
        assert torch.allclose(var.default_value, default)

    def test_creation_with_dtype(self):
        """Test dtype with torch dtype object."""
        var = TorchNDVariable(name="test", shape=(10,), dtype=torch.float64)
        assert var.dtype == torch.float64

    def test_creation_with_int_dtype(self):
        """Test int dtype is accepted."""
        var = TorchNDVariable(name="test", shape=(10,), dtype=torch.int32)
        assert var.dtype == torch.int32

    def test_invalid_dtype_type_raises(self):
        """Test invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be a"):
            TorchNDVariable(name="test", shape=(10,), dtype="invalid")

    # Value validation tests
    def test_validate_value_correct_tensor(self):
        """Test validation of correct tensor value."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        var.validate_value(torch.randn(10, 20))

    def test_validate_value_batched_tensor(self):
        """Test validation of batched tensor should fail."""
        with pytest.raises(ValueError, match="Expected shape"):
            var = TorchNDVariable(name="test", shape=(10, 20))
            var.validate_value(torch.randn(5, 10, 20))

    def test_validate_value_wrong_type_raises(self):
        """Test that non-tensor value raises TypeError."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        with pytest.raises(TypeError, match="Expected value to be a Tensor"):
            var.validate_value([[1, 2], [3, 4]])

    def test_validate_value_wrong_shape_raises(self):
        """Test that wrong shape raises ValueError."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        with pytest.raises(ValueError, match="Expected shape"):
            var.validate_value(torch.randn(10, 30))

    def test_validate_value_wrong_dtype_raises(self):
        """Test that wrong dtype raises ValueError."""
        var = TorchNDVariable(name="test", shape=(10,), dtype=torch.float32)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(torch.randn(10, dtype=torch.float64))

    # Read-only validation tests
    # Note: read-only validation is handled by _validate_read_only, not
    # validate_value. validate_value only checks type/dtype/range/shape.
    # The model layer is responsible for calling _validate_read_only separately.
    def test_read_only_matching_value(self):
        """Test read-only ND variable with matching value."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        var._validate_read_only(default.clone())

    def test_read_only_non_matching_value_raises(self):
        """Test read-only ND variable with non-matching value raises."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        with pytest.raises(ValueError, match="read-only"):
            var._validate_read_only(torch.randn(10, 20))

    def test_read_only_no_default_raises(self):
        """Test read-only ND variable without default raises."""
        var = TorchNDVariable(name="test", shape=(10, 20), read_only=True)
        with pytest.raises(ValueError, match="no default value"):
            var._validate_read_only(torch.randn(10, 20))

    def test_read_only_value_within_tolerance(self):
        """Test read-only ND variable with values within tolerance."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        # Value very close to default (within tolerance)
        close_value = default + 1e-10
        var._validate_read_only(close_value)


class TestDistributionVariable:
    """Tests for DistributionVariable class."""

    def test_basic_creation(self):
        """Test basic distribution variable creation."""
        var = DistributionVariable(name="dist_var")
        assert var.name == "dist_var"
        assert var.unit is None

    def test_missing_name_raises_validation_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            DistributionVariable(unit="meters")

    def test_creation_with_unit(self):
        """Test distribution variable with unit."""
        var = DistributionVariable(name="dist_var", unit="meters")
        assert var.unit == "meters"

    def test_validate_normal_distribution(self):
        """Test validation of Normal distribution."""
        var = DistributionVariable(name="test")
        dist = Normal(loc=0.0, scale=1.0)
        var.validate_value(dist)

    def test_validate_uniform_distribution(self):
        """Test validation of Uniform distribution."""
        var = DistributionVariable(name="test")
        dist = Uniform(low=0.0, high=1.0)
        var.validate_value(dist)

    def test_validate_non_distribution_raises(self):
        """Test that non-distribution value raises TypeError."""
        var = DistributionVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value(5.0)

    def test_validate_tensor_raises(self):
        """Test that tensor value raises TypeError."""
        var = DistributionVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value(torch.tensor([1.0, 2.0]))


class TestConfigEnum:
    """Tests for ConfigEnum."""

    def test_enum_values(self):
        """Test ConfigEnum values."""
        assert ConfigEnum.NULL.value == "none"
        assert ConfigEnum.WARN.value == "warn"
        assert ConfigEnum.ERROR.value == "error"

    def test_enum_from_string(self):
        """Test ConfigEnum creation from string."""
        assert ConfigEnum("none") == ConfigEnum.NULL
        assert ConfigEnum("warn") == ConfigEnum.WARN
        assert ConfigEnum("error") == ConfigEnum.ERROR


class TestVariableInheritance:
    """Tests for variable inheritance structure."""

    def test_torch_scalar_variable_is_variable(self):
        """Test TorchScalarVariable is a Variable."""
        var = TorchScalarVariable(name="test")
        assert isinstance(var, Variable)

    def test_torch_nd_variable_is_variable(self):
        """Test TorchNDVariable is a Variable."""
        var = TorchNDVariable(name="test", shape=(10,))
        assert isinstance(var, Variable)

    def test_distribution_variable_is_variable(self):
        """Test DistributionVariable is a Variable."""
        var = DistributionVariable(name="test")
        assert isinstance(var, Variable)


class TestTorchNDVariableEdgeCases:
    """Additional edge case tests for TorchNDVariable."""

    def test_1d_shape(self):
        """Test 1D shape variable."""
        var = TorchNDVariable(name="test", shape=(100,))
        var.validate_value(torch.randn(100))

    def test_4d_shape(self):
        """Test 4D shape variable (e.g., video data)."""
        var = TorchNDVariable(name="test", shape=(10, 3, 64, 64))
        var.validate_value(torch.randn(10, 3, 64, 64))

    def test_model_dump_nd_variable(self):
        """Test model_dump for TorchNDVariable."""
        var = TorchNDVariable(name="test", shape=(10, 20), unit="pixels")
        dump = var.model_dump()
        assert dump["variable_class"] == "TorchNDVariable"
        assert dump["name"] == "test"
        assert dump["shape"] == (10, 20)
        assert dump["unit"] == "pixels"

    def test_default_value_wrong_shape_raises(self):
        """Test that default value with wrong shape raises error."""
        with pytest.raises(ValueError, match="Expected shape"):
            TorchNDVariable(
                name="test", shape=(10, 20), default_value=torch.randn(10, 30)
            )

    def test_default_value_wrong_dtype_raises(self):
        """Test that default value with wrong dtype raises error."""
        with pytest.raises(ValueError, match="Expected dtype"):
            TorchNDVariable(
                name="test",
                shape=(10,),
                dtype=torch.float32,
                default_value=torch.randn(10, dtype=torch.float64),
            )

    def test_validate_value_config_parameter(self):
        """Test validate_value with config parameter."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        # Should work with config parameter (even though optional validation is not implemented)
        var.validate_value(torch.randn(10, 20), config="error")
        var.validate_value(torch.randn(10, 20), config="warn")
        var.validate_value(torch.randn(10, 20), config="none")


class TestDistributionVariableEdgeCases:
    """Additional edge case tests for DistributionVariable."""

    def test_read_only_attribute(self):
        """Test read_only attribute on distribution variable."""
        var = DistributionVariable(name="test", read_only=True)
        assert var.read_only is True

    def test_default_validation_config(self):
        """Test default_validation_config attribute."""
        var = DistributionVariable(name="test", default_validation_config="warn")
        assert var.default_validation_config == ConfigEnum.WARN

    def test_model_dump(self):
        """Test model_dump for DistributionVariable."""
        var = DistributionVariable(name="test", unit="meters")
        dump = var.model_dump()
        assert dump["variable_class"] == "DistributionVariable"
        assert dump["name"] == "test"
        assert dump["unit"] == "meters"

    def test_validate_with_config_parameter(self):
        """Test validate_value with config parameter."""
        var = DistributionVariable(name="test")
        dist = Normal(loc=0.0, scale=1.0)
        var.validate_value(dist, config="error")
        var.validate_value(dist, config=ConfigEnum.WARN)

    def test_validate_batched_distribution(self):
        """Test validation of batched distribution."""
        var = DistributionVariable(name="test")
        # Batched normal distribution
        dist = Normal(loc=torch.zeros(5), scale=torch.ones(5))
        var.validate_value(dist)

    def test_validate_multivariate_distribution(self):
        """Test validation of multivariate distribution."""
        from torch.distributions import MultivariateNormal

        var = DistributionVariable(name="test")
        dist = MultivariateNormal(
            loc=torch.zeros(3),
            covariance_matrix=torch.eye(3),
        )
        var.validate_value(dist)


# TODO: REVIEW and DELETE LATER


class TestTorchNDVariableCreation:
    def test_creation_defaults(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        assert var.name == "t"
        assert var.shape == (3, 4)
        assert var.dtype == torch.float32
        assert var.default_value is None

    def test_creation_custom_dtype(self):
        var = TorchNDVariable(name="t", shape=(2,), dtype=torch.float64)
        assert var.dtype == torch.float64

    def test_creation_with_default_value(self):
        t = torch.ones(3, 4)
        var = TorchNDVariable(name="t", shape=(3, 4), default_value=t)
        assert torch.equal(var.default_value, t)

    def test_unit_field(self):
        var = TorchNDVariable(name="t", shape=(2,), unit="m/s")
        assert var.unit == "m/s"


class TestTorchNDVariableValidation:
    def test_validate_correct_tensor(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        t = torch.zeros(3, 4)
        var.validate_value(t, config="error")  # must not raise

    def test_validate_1d(self):
        var = TorchNDVariable(name="t", shape=(5,))
        var.validate_value(torch.zeros(5), config="error")

    def test_wrong_shape_raises(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        with pytest.raises(ValueError, match="Expected shape"):
            var.validate_value(torch.zeros(3, 5), config="error")

    def test_wrong_dtype_raises(self):
        var = TorchNDVariable(name="t", shape=(3, 4), dtype=torch.float64)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(torch.zeros(3, 4, dtype=torch.float32), config="error")

    def test_reject_ndarray(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        with pytest.raises(TypeError, match="Expected value to be a Tensor"):
            var.validate_value(np.zeros((3, 4)), config="error")

    def test_reject_list(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        with pytest.raises(TypeError, match="Expected value to be a Tensor"):
            var.validate_value([[1, 2, 3, 4]] * 3, config="error")

    def test_default_value_validated_on_creation(self):
        """Wrong dtype in default_value must be caught at construction time."""
        with pytest.raises((ValueError, Exception)):
            TorchNDVariable(
                name="t",
                shape=(2,),
                dtype=torch.float64,
                default_value=torch.zeros(2, dtype=torch.float32),
            )


class TestTorchNDVariableDtypeCoercion:
    def test_string_dtype_coerced(self):
        var = TorchNDVariable(name="t", shape=(2,), dtype="float32")
        assert var.dtype == torch.float32

    def test_string_dtype_with_prefix_coerced(self):
        """Strings like 'torch.float64' (from old serializations) are accepted."""
        var = TorchNDVariable(name="t", shape=(2,), dtype="torch.float64")
        assert var.dtype == torch.float64

    def test_invalid_string_dtype_raises(self):
        with pytest.raises(TypeError):
            TorchNDVariable(name="t", shape=(2,), dtype="not_a_dtype")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            TorchNDVariable(name="t", shape=(2,), dtype=42)


class TestTorchNDVariableSerialization:
    def test_dtype_serialized_as_string(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        dumped = var.model_dump()
        assert dumped["dtype"] == "torch.float32"
        assert isinstance(dumped["dtype"], str)

    def test_dtype_float32_serialized(self):
        var = TorchNDVariable(name="t", shape=(3, 4), dtype=torch.float32)
        assert var.model_dump()["dtype"] == "torch.float32"

    def test_default_value_none_serialized(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        assert var.model_dump()["default_value"] is None

    def test_default_value_serialized_as_list(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)  ### ????
        var = TorchNDVariable(name="t", shape=(3, 4), default_value=t)
        dumped = var.model_dump()
        assert dumped["default_value"] == t.tolist()
        assert isinstance(dumped["default_value"], list)

    def test_variable_class_in_dump(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        assert var.model_dump()["variable_class"] == "TorchNDVariable"

    def test_model_dump_is_json_serializable(self):
        t = torch.ones(3, 4)
        var = TorchNDVariable(name="t", shape=(3, 4), default_value=t)
        dumped = var.model_dump()
        # Must not raise
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert parsed["dtype"] == "torch.float32"
        assert parsed["default_value"] == t.tolist()

    def test_json_serializable_no_default(self):
        var = TorchNDVariable(name="t", shape=(5,), dtype=torch.int32)
        json_str = json.dumps(var.model_dump())
        parsed = json.loads(json_str)
        assert parsed["dtype"] == "torch.int32"
        assert parsed["default_value"] is None


class TestTorchNDVariableRoundTrip:
    def test_roundtrip_no_default_value(self):
        var = TorchNDVariable(name="weights", shape=(3, 4), dtype=torch.float32)
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        parsed.pop("variable_class", None)

        var2 = TorchNDVariable(**parsed)
        assert var2.name == var.name
        assert var2.shape == var.shape
        assert var2.dtype == var.dtype
        assert var2.default_value is None

    def test_roundtrip_with_default_value(self):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)  ###??
        var = TorchNDVariable(name="t", shape=(2, 3), default_value=t)
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)

        var2 = TorchNDVariable(**parsed)
        assert var2.name == var.name
        assert var2.shape == var.shape
        assert var2.dtype == var.dtype
        assert torch.equal(var2.default_value, t)

    def test_roundtrip_integer_dtype(self):
        t = torch.arange(4, dtype=torch.int64)
        var = TorchNDVariable(
            name="idx", shape=(4,), dtype=torch.int64, default_value=t
        )
        parsed = json.loads(json.dumps(var.model_dump()))

        var2 = TorchNDVariable(**parsed)
        assert var2.dtype == torch.int64
        assert torch.equal(var2.default_value, t)

    def test_roundtrip_double_dtype(self):
        t = torch.arange(4, dtype=torch.double)
        var = TorchNDVariable(name="v", shape=(4,), dtype=torch.double, default_value=t)
        parsed = json.loads(json.dumps(var.model_dump()))

        var2 = TorchNDVariable(**parsed)
        assert var2.dtype == torch.double
        assert torch.equal(var2.default_value, t)

    def test_roundtrip_1d(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        var = TorchNDVariable(
            name="v", shape=(3,), dtype=torch.float64, default_value=t
        )
        parsed = json.loads(json.dumps(var.model_dump()))

        var2 = TorchNDVariable(**parsed)
        assert var2.dtype == torch.float64
        assert torch.equal(var2.default_value, t)

    def test_roundtrip_unit_preserved(self):
        var = TorchNDVariable(name="v", shape=(2,), unit="GeV")
        parsed = json.loads(json.dumps(var.model_dump()))
        parsed.pop("variable_class", None)

        var2 = TorchNDVariable(**parsed)
        assert var2.unit == "GeV"
