"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.
"""

import logging
import warnings
from typing import Optional, Type, Union, ClassVar, List, Any, Self

import torch
from torch import Tensor
from torch.distributions import Distribution as TDistribution
from pydantic import (
    field_validator,
    model_validator,
    ConfigDict,
    Field,
    field_serializer,
)

from lume.variables import Variable, ScalarVariable, NDVariable, ConfigEnum

logger = logging.getLogger(__name__)

# Rename base ScalarVariable for internal use
_BaseScalarVariable = ScalarVariable

# Re-export base classes for backward compatibility and clean API
# We alias TorchScalarVariable (defined below) as ScalarVariable for backwards compatibility
# with a deprecation warning. See end of this module for the aliasing.
# NOTE: ScalarVariable will be deprecated in the next release - use TorchScalarVariable instead.
__all__ = [
    "Variable",
    "ScalarVariable",
    "TorchScalarVariable",
    "NDVariable",
    "TorchNDVariable",
    "ConfigEnum",
    "DistributionVariable",
    "get_variable",
]

# Canonical map from string name → torch.dtype.
# Keys are what serialize_dtype produces (str(dtype).replace("torch.", "")).
# Aliases (e.g. "double" → float64) intentionally omitted so the map is
# bijective and round-trips cleanly.
_STR_TO_TORCH_DTYPE: dict[str, "torch.dtype"] = {
    str(d).replace("torch.", ""): d
    for d in [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ]
}


class DistributionVariable(Variable):
    """Variable for distributions. Must be a subclass of torch.distributions.Distribution.

    Attributes
    ----------
    unit : str, optional
        Unit associated with the variable.

    """

    unit: Optional[str] = None

    def validate_value(
        self, value: TDistribution, config: Optional[ConfigEnum] = None, **kwargs
    ):
        """Validates the given value.

        Parameters
        ----------
        value : Distribution
            The value to be validated.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not an instance of Distribution.

        """
        # mandatory validation
        self._validate_value_type(value)

        # optional validation
        config = self._validation_config_as_enum(config)
        if config != ConfigEnum.NULL:
            pass  # not implemented

    @staticmethod
    def _validate_value_type(value: TDistribution):
        if not isinstance(value, TDistribution):
            raise TypeError(
                f"Expected value to be of type {TDistribution}, "
                f"but received {type(value)}."
            )


class TorchScalarVariable(_BaseScalarVariable):
    """Variable for scalar values represented as PyTorch tensors.

    This class extends ScalarVariable to support scalar values as torch.Tensor
    with 0 or 1 dimensions (i.e., a scalar tensor or a single-element tensor).

    Attributes
    ----------
    default_value : Tensor | None
        Default value for the variable (must be 0D or 1D with size 1).
    dtype : torch.dtype | None
        Optional data type of the tensor. If specified, validates that tensor values
        match this exact dtype. If None (default), only validates that the dtype is
        a floating-point type without enforcing a specific precision.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_value: Optional[Union[Tensor, float]] = None
    dtype: Optional[torch.dtype] = None

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        """Validate that dtype is a torch.dtype and is a floating-point type."""
        if value is None:
            return None

        # Validate that value is a torch.dtype instance
        if not isinstance(value, torch.dtype):
            raise TypeError(
                f"dtype must be a torch.dtype instance, "
                f"got {type(value).__name__}. "
                f"Received value: {repr(value)}"
            )

        # Validate that the dtype is a floating-point type
        if not value.is_floating_point:
            raise ValueError(f"dtype must be a floating-point type, got {value}")
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def validate_value(
        self, value: Union[Tensor, float], config: Optional[ConfigEnum] = None
    ):
        """Validates the given tensor or float value.

        Performs type, dtype, and range validation. Read-only validation
        is not performed here — it is the responsibility of the model layer
        to call ``_validate_read_only`` separately when appropriate.

        Parameters
        ----------
        value : Tensor | float
            The value to be validated. If a tensor, must be a 0D scalar tensor.
            Batched tensors must be unbatched at the model layer before calling
            this method.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not a torch.Tensor or float.
        ValueError
            If a tensor is not a 0D scalar, or if tensor dtype is not a float
            type, or if value is out of range.

        """
        # mandatory validation
        self._validate_value_type(value)
        self._validate_dtype(value)

        # optional validation
        config = self._validation_config_as_enum(config)

        if config != ConfigEnum.NULL:
            scalar_value = value.item() if isinstance(value, Tensor) else value
            self._validate_value_is_within_range(scalar_value, config=config)

    def _validate_value_type(self, value):
        """Validates that value is a 0D torch.Tensor or a regular float/int.

        Batched tensors are not accepted here; batch handling is the
        responsibility of the model layer
        (see :meth:`LUMETorch._validate_dict_per_variable`).
        """
        if isinstance(value, Tensor):
            if value.ndim != 0:
                raise ValueError(
                    f"Expected a 0D scalar tensor, "
                    f"but got {value.ndim} dimensions with shape {value.shape}. "
                    f"Batched tensors should be unbatched at the model layer."
                )
        else:
            # Delegate to parent class for non-tensor validation
            _BaseScalarVariable._validate_value_type(value)

    def _validate_dtype(self, value):
        """Validates the dtype of the tensor is a float type. Skips check for regular floats."""
        if not isinstance(value, Tensor):
            return  # Regular floats don't have dtype to validate
        if not value.dtype.is_floating_point:
            raise ValueError(
                f"Expected tensor dtype to be a floating-point type, got {value.dtype}."
            )
        if self.dtype and value.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {value.dtype}")

    def _validate_read_only(self, value: Union[Tensor, float]):
        """Validates that a read-only variable's value matches its default.

        This method validates a single scalar value (float or 0D/single-element
        tensor). Batch handling is the responsibility of the model layer.
        """
        if not self.read_only:
            return

        if self.default_value is None:
            raise ValueError(
                f"Variable '{self.name}' is read-only but has no default value."
            )

        # Extract scalar value from default if it's a tensor
        if isinstance(self.default_value, Tensor):
            expected_scalar = self.default_value.item()
        else:
            expected_scalar = self.default_value

        # Extract scalar value from input
        if isinstance(value, Tensor):
            scalar_value = value.item()
        else:
            scalar_value = value

        if abs(expected_scalar - scalar_value) >= 1e-9:
            raise ValueError(
                f"Variable '{self.name}' is read-only and must equal its default value "
                f"({expected_scalar}), but received {scalar_value}."
            )


class TorchNDVariable(NDVariable):
    """Variable for PyTorch tensor data.

    Attributes
    ----------
    default_value : Tensor | None
        Default value for the variable. Must match the expected
        shape and dtype if provided. Defaults to None.
    dtype : torch.dtype
        Expected element type. Defaults to torch.float64.
    default_value : torch.Tensor | None
        Optional default tensor. Validated against shape and dtype.

    Examples
    --------
    >>> import torch
    >>> from lume_torch.variables import TorchNDVariable
    >>>
    >>> var = TorchNDVariable(
    ...     name="my_tensor",
    ...     shape=(3, 4),
    ...     dtype=torch.float32
    ... )
    >>>
    >>> tensor = torch.rand(3, 4)
    >>> var.validate_value(tensor, config="error")  # Passes

    """

    default_value: Optional[Tensor] = None
    dtype: torch.dtype = torch.float32

    array_type: ClassVar[type] = Tensor

    # Internal flag: set to True when default_value was coerced from a
    # list so validate_default_value knows to cast rather than reject.
    # Excluded from serialization; populated via _mark_coerced_default_value.
    default_value_was_coerced: bool = Field(default=False, exclude=True, repr=False)

    @classmethod
    def _dtype_coerce(cls, value: Any) -> "torch.dtype":
        """Coerce a string (e.g. "float64") to the matching torch.dtype.

        Parameters
        ----------
        value : Any
            Typically a string produced by serialize_dtype.

        Returns
        -------
        torch.dtype
            The corresponding PyTorch dtype.

        Raises
        ------
        TypeError
            If value is not a string or does not map to a known dtype.

        """
        if not isinstance(value, str):
            raise TypeError(
                f"Cannot coerce {type(value).__name__!r} to torch.dtype; "
                "expected a string."
            )
        key = value.replace("torch.", "")  # tolerate "torch.float64" too
        try:
            return _STR_TO_TORCH_DTYPE[key]
        except KeyError:
            raise TypeError(
                f"Cannot coerce {value!r} to torch.dtype. "
                f"Known names: {sorted(_STR_TO_TORCH_DTYPE)}"
            )

    @field_serializer("dtype")
    def serialize_dtype(self, value: "torch.dtype") -> str:
        """Serialize torch.dtype to its full string representation.

        Parameters
        ----------
        value : torch.dtype
            The dtype to serialize.

        Returns
        -------
        str
            Dtype string, e.g. "torch.float32", "torch.int64".

        """
        return str(value)

    @field_serializer("default_value")
    def serialize_default_value(
        self, value: Optional["torch.Tensor"]
    ) -> Optional[List]:
        """Serialize a torch.Tensor to a nested Python list.

        Parameters
        ----------
        value : torch.Tensor or None

        Returns
        -------
        list or None

        """
        if value is None:
            return None
        return value.tolist()

    @model_validator(mode="before")
    @classmethod
    def _mark_coerced_default_value(cls, data: Any) -> Any:
        """Detect list default_value input before field validation.

        When default_value arrives as a list, tuple (e.g.
        after a JSON round-trip), Pydantic cannot infer the intended
        torch.dtype. This validator sets a _default_value_was_coerced
        key in the raw dict so that validate_default_value knows to
        cast the resulting tensor to self.dtype rather than reject the
        dtype mismatch.

        """
        if isinstance(data, dict):
            dv = data.get("default_value")
            if isinstance(dv, (list, tuple)):
                data = dict(data)
                data["default_value_was_coerced"] = True
        return data

    @field_validator("default_value", mode="before")
    @classmethod
    def coerce_default_value(cls, value: Any) -> Any:
        """Coerce a list/tuple back to a torch.Tensor on round-trip.

        The tensor is created with PyTorch's inferred dtype; the correct dtype
        is applied later in validate_default_value once self.dtype is known.

        Parameters
        ----------
        value : Any
            Raw input. Lists and tuples are converted via torch.tensor(value).

        Returns
        -------
        torch.Tensor or None or Any

        """
        if isinstance(value, (list, tuple)):
            return torch.tensor(value)
        return value

    @model_validator(mode="after")
    def validate_default_value(self) -> Self:
        """Cast or validate default_value, then run base shape/type checks.

        * If default_value was coerced from a list (flagged by
          _mark_coerced_default_value), it is cast to self.dtype so the
          round-trip works without the caller having to specify the dtype twice.
        * If it was supplied as an explicit torch.Tensor with a mismatched
          dtype, the base-class validator raises ValueError as expected.

        Returns
        -------
        TorchNDVariable
            The validated instance.

        """
        if (
            self.default_value is not None
            and self.default_value_was_coerced
            and self.default_value.dtype != self.dtype
        ):
            object.__setattr__(self, "default_value", self.default_value.to(self.dtype))
        # Reset flag for subsequent validate_assignment calls
        object.__setattr__(self, "default_value_was_coerced", False)
        # Delegate shape / type checks to base
        return super().validate_default_value()

    def _validate_read_only(self, value: Tensor) -> None:
        """Validates that a read-only ND-variable's value matches its default.

        This method validates a single unbatched tensor. Batch handling is
        the responsibility of the model layer.
        """
        if not self.read_only:
            return

        if self.default_value is None:
            raise ValueError(
                f"Variable '{self.name}' is read-only but has no default value."
            )

        if not torch.allclose(value, self.default_value, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"Variable '{self.name}' is read-only and must equal its default value, "
                f"but received different array values."
            )

    def validate_value(self, value: Tensor, config: str = None):
        """Validates the given tensor value.

        Performs type, dtype, and shape validation. Read-only validation
        is not performed here — it is the responsibility of the model layer
        to call ``_validate_read_only`` separately when appropriate.

        Parameters
        ----------
        value : Tensor
            The tensor value to validate.
        config : str, optional
            The validation configuration.

        """
        super().validate_value(value, config)


# Alias TorchScalarVariable as ScalarVariable for backwards compatibility
# This will be deprecated in the next release
class ScalarVariable(TorchScalarVariable):
    """Deprecated alias for TorchScalarVariable.

    .. deprecated::
        ScalarVariable is deprecated and will be removed in the next release.
        Use TorchScalarVariable instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ScalarVariable is deprecated and will be removed in the next release. "
            "Please use TorchScalarVariable instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def get_variable(name: str) -> Type[Variable]:
    """Returns the Variable subclass with the given name.

    Parameters
    ----------
    name : str
        Name of the Variable subclass.

    Returns
    -------
    Type[Variable]
        Variable subclass with the given name.

    """
    classes = [
        TorchScalarVariable,
        ScalarVariable,
        DistributionVariable,
        TorchNDVariable,
    ]
    class_lookup = {c.__name__: c for c in classes}
    if name not in class_lookup.keys():
        logger.error(
            f"Unknown variable type '{name}', valid names are {list(class_lookup.keys())}"
        )
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]
