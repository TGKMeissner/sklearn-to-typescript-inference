# sklearn-to-ts

A Claude Code plugin that trains sklearn models in Python, exports weights as JSON, and generates pure TypeScript inference code — no Python runtime, no ONNX, no TensorFlow.js. Zero npm dependencies. The resulting model runs server-side (Node.js/Next.js) or client-side (browser) in <1ms with a ~10-50 KB weights file.

## Install

```bash
claude plugin marketplace add TGKMeissner/sklearn-to-typescript-inference
claude plugin install sklearn-to-ts
```

## What it does

This skill guides the full ML-to-production pipeline:

1. **Design features** — helps you identify and engineer the right input features for your prediction task
2. **Generate training data** — creates synthetic datasets using domain formulas + noise, or works with your existing data
3. **Train an sklearn model** — MLP, Random Forest, Linear Regression, Ridge, Lasso, or Gradient Boosting
4. **Export weights to JSON** — extracts model weights, biases, and normalization parameters into a portable JSON file
5. **Generate TypeScript inference** — produces a zero-dependency forward pass function with feature extraction, imputation, and sanity clamping
6. **Integrate into your app** — wires the model into your existing codebase with proper types and error handling

The entire pipeline runs locally. No external APIs, no cloud training, no model hosting.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  TRAINING (Python, one-time)                    │
│                                                 │
│  Raw Data → Feature Engineering → sklearn Model │
│                              ↓                  │
│                     export-weights.py           │
│                              ↓                  │
│                     model-weights.json          │
└─────────────────────────────────────────────────┘
                       ↓ committed to repo
┌─────────────────────────────────────────────────┐
│  INFERENCE (TypeScript, runtime)                │
│                                                 │
│  Raw Input → extractFeatures() → forwardPass()  │
│                                     ↓           │
│                              Prediction          │
└─────────────────────────────────────────────────┘
```

## Example use cases

### Fitness & Sports

| Use case | Features | Prediction |
|----------|----------|------------|
| **Race time prediction** | Interval pace, tempo pace, long run distance, weekly volume, HR ratios, pace trend | 5K, 10K, Half Marathon, Marathon finish times |
| **VO2max estimation** | Interval pace at high HR, easy pace at low HR, HR recovery rate, long run HR drift | VO2max (ml/kg/min) without a lab test |
| **Training zone prediction** | Pace + HR at various intensities, pace-HR coupling trend | Personalized lactate threshold and training zones |
| **Recovery & readiness** | Acute/chronic training load ratio, days since rest, RPE trend, resting HR | Readiness score (0-100) |
| **Injury risk scoring** | Volume ramp rate, training monotony, consecutive hard days, muscle group imbalance | Risk score or 30-day injury probability |
| **Strength progression** | Per-lift history (weight × reps × sets), RPE, session frequency, body weight | Predicted 1RM, recommended working weight |

### SaaS & Product

| Use case | Features | Prediction |
|----------|----------|------------|
| **Churn prediction** | Login frequency, feature usage depth, support tickets, days since last activity | Churn probability within 30 days |
| **Lead scoring** | Company size, industry, page visits, email opens, demo requests | Conversion likelihood (0-100) |
| **Usage forecasting** | Historical API calls, user growth rate, seasonal patterns | Projected resource usage for capacity planning |

### E-commerce

| Use case | Features | Prediction |
|----------|----------|------------|
| **Price optimization** | Competitor prices, demand history, inventory level, seasonality | Optimal price point |
| **Delivery time estimation** | Distance, warehouse load, carrier performance history, weather | Estimated delivery window |
| **Return probability** | Product category, price, customer history, review sentiment | Likelihood of return |

### Content & Media

| Use case | Features | Prediction |
|----------|----------|------------|
| **Content performance** | Title length, topic category, publish time, author history | Expected engagement score |
| **Read time estimation** | Word count, sentence complexity, image count, code blocks | Accurate read time in minutes |
| **Recommendation scoring** | User interaction history, content similarity, recency | Relevance score for ranking |

## Supported model types

- **MLP (Multi-Layer Perceptron)** — best for continuous regression and multi-output prediction
- **Linear Regression / Ridge / Lasso** — simplest case, single matrix multiply
- **Random Forest / Gradient Boosting** — exported as decision tree arrays with tree traversal in TypeScript

## Usage

The skill is automatically invoked when you ask Claude to train and deploy an ML model in TypeScript. You can also invoke it directly:

```
/sklearn-to-ts Train an MLP to predict 5K race times from training log features
```

```
/sklearn-to-ts Build a churn predictor from user activity data, deploy as TypeScript
```

```
/sklearn-to-ts Export my existing sklearn Random Forest to a zero-dependency TS function
```

## When NOT to use this

- **Image/audio/video models** — use ONNX runtime or TensorFlow.js
- **Models > 1MB of weights** — consider a server-side inference API
- **Models requiring GPU** — this is CPU-only
- **Frequently retrained models** (daily) — consider a model serving API
- **Deep learning (>3 layers)** — the hand-rolled forward pass works but gets unwieldy

## Real-world example

This plugin was born from [Hypla](https://hypla.fit), a hybrid training tracker that uses a distilled MLP to predict race times from logged workouts. The model is ~15 KB of JSON weights, runs client-side in <1ms, and updates predictions in real-time as athletes log new sessions.

## License

MIT
