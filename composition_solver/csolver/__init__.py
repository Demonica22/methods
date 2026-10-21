"""Решатель ОДУ на композиционных схемах с комплексными коэффициентами."""
from .base_methods import Trapezoid, KD, GraggMidpoint, BASE_METHODS
from .compositions import Composition, SCHEMES, triple_jump_gamma, found_schemes
from .solver import integrate
from . import problems
from .adaptive import EmbeddedPair, PAIRS, integrate_adaptive
