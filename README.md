# sklearn-to-ts

Claude Code plugin for training sklearn models and running inference in pure TypeScript — no Python runtime, no ONNX, no TensorFlow.js. Zero npm dependencies.

## Install

```bash
claude plugin marketplace add TGKMeissner/sklearn-to-typescript-inference
claude plugin install sklearn-to-ts
```

## What it does

1. **Train** an sklearn model (MLP, Random Forest, Linear Regression, etc.) in Python
2. **Export** the learned weights as a JSON file
3. **Generate** a TypeScript inference function that runs the model with zero dependencies

The skill handles the full pipeline: synthetic data generation, model training, weight export, and TypeScript codegen.

## Use cases

- Deploy ML models in JavaScript/TypeScript apps without a Python backend
- Browser-based inference (no server round-trips)
- Lightweight prediction models (race times, VO2max, recovery, injury risk)

## Usage

Once installed, the skill is automatically invoked when you ask Claude to train and deploy an ML model in TypeScript. You can also invoke it directly:

```
/sklearn-to-ts Train an MLP to predict 5K race times from training features
```

## License

MIT
