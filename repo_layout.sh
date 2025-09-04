#!/bin/bash

# Create subdirectories
mkdir -p .github/workflows
mkdir -p src
mkdir -p tests

# Create files in .github/workflows
touch .github/workflows/ci.yml

# Create files in src
touch src/config.py
touch src/control_nets.py
touch src/fhn.py
touch src/train.py
touch src/viz.py

# Create files in tests
touch tests/test_integrator.py

# Create other files in the main directory
touch .gitignore
touch LICENSE
touch Makefile
touch pyproject.toml

echo "Directory structure created successfully."