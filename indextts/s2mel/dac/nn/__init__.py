"""Inference-safe DAC neural network namespace.

Submodules are imported explicitly by their consumers.  Avoid importing the
training-only loss module here because it pulls in descript-audiotools.
"""
