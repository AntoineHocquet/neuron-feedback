"""
Tests for integrators in fhn.py
"""
from src.fhn import simulate_no_grad, simulate_grad, FHNVectorField
import pytest

@pytest.fixture
def fhn_vector_field() -> FHNVectorField:
    return FHNVectorField()