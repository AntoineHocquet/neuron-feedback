# FitzHugh Feedback Control

[![CI](https://github.com/your-username/fhn-feedback-control/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/fhn-feedback-control/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Closed-loop feedback control of the **FitzHugh–Nagumo model** with a learned feedback law  
\\( I(t) = f_\theta(v(t)) \\), trained to minimize a quadratic tracking cost

\\[
J = \int_0^T (v(t) - v_{\text{ref}}(t))^2 \, dt .
\\]

This repo demonstrates modern **ML + controls** practice:  
- differentiable ODE integration,  
- neural or polynomial controllers with input saturation,  
- config-first design (CLI overrides),  
- clean MLOps hygiene (tests, CI, reproducibility).

---

## 📂 Repository Layout

fhn_feedback_control/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── config.py
│   ├── control_nets.py
│   ├── fhn.py
│   ├── train.py
│   └── viz.py
├── tests/
│   └── test_integrator.py
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml



---

## ⚡ Quickstart

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Train with defaults
python -m src.train

All outputs go into:

runs/metrics.csv → training loss log

outputs/*.png → plots (state, control, reference)


## 🔧 Config Overrides

Every hyperparameter can be overridden from the CLI using dotted keys:
# longer horizon, stronger input bound
python -m src.train sim.T=40.0 ctrl.umax=2.5

# polynomial controller instead of MLP
python -m src.train ctrl.kind=poly

# sinusoidal reference with custom amplitude/frequency
python -m src.train ref.kind=sinus ref.amp=1.5 ref.freq=0.25

📊 Example Results

After training, you’ll get plots like:

Voltage vs Reference


Recovery Variable


Control Input


## 🧠 Features

Differentiable RK4 integrator (PyTorch autograd-friendly)

Multiple feedback laws: MLP or polynomial (with tanh saturation)

Reference generators: constant, step, sinusoidal (or plug in CSV)

Safe training: gradient clipping, bounded inputs

Reproducible: seeded RNG, config-based, CI-tested

## ✅ Tests & CI

make lint     # ruff lint
make test     # pytest

CI runs automatically on push/PR:

Linting (ruff)

Unit test (gradient-flow through rollout)

## 📌 Roadmap
Add quadratic control penalty (\( \lambda \int I^2 dt \))
TensorBoard / MLflow logging
Adaptive ODE solvers (torchdiffeq)
Dockerfile for full reproducibility

## 📜 License

MIT
