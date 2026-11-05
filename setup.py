#!/usr/bin/env python3
"""
Setup configuration for AgyMolden.
Standard setuptools installer compatible with pip, sdist, and bdist_wheel.
"""
from setuptools import setup, find_packages

setup(
    name="agymolden",
    version="1.0.0",
    description="AgyMolden - 3D Z-Matrix molecular builder & quantum chemistry viewer",
    author="AgyMolden Developers",
    packages=find_packages(include=["agymolden*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "PyQt6>=6.4",
        "PyOpenGL>=3.1",
        "numpy>=1.23",
        "scipy>=1.9",
        "matplotlib>=3.5",
        "scikit-image>=0.19",
        "cclib>=1.7",
    ],
    extras_require={
        "all": [
            "rdkit>=2022.03",
            "basis_set_exchange>=0.9",
            "pyscf>=2.0; sys_platform != 'win32'",
        ],
        "qm": [
            "pyscf>=2.0; sys_platform != 'win32'",
            "cclib>=1.7",
            "basis_set_exchange>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "agymolden=agymolden:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
