---
name: sklearn-to-ts
description: Train an sklearn model (MLP, Random Forest, etc.), export weights to JSON, and run inference in pure TypeScript with zero dependencies. Use when the user wants to deploy a machine learning model in a JavaScript/TypeScript app without Python runtime, ONNX, or TensorFlow.js. Also use when the user mentions "ML in the browser", "model inference in TypeScript", "export sklearn model", "deploy ML without Python", "lightweight ML inference", "train and deploy model", "neural network in TypeScript", "predict VO2 max", "training zone prediction", "fitness prediction model", "recovery prediction", or "injury risk model".
---

# sklearn → TypeScript Inference

Train a machine learning model in Python (sklearn), export the weights as JSON, and run inference in pure TypeScript — no Python runtime, no ONNX, no TensorFlow.js. Zero npm dependencies.

## When to Use This Skill

- Deploying an ML model in a Next.js / Node.js / browser app
- Need inference without a Python backend or heavy ML runtime
- Want a lightweight, dependency-free prediction pipeline
- Exporting sklearn MLP, Random Forest, or linear models to TypeScript
- Building real-time predictions that run server-side or client-side in JS/TS

## Architecture Overview

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

## Step 1: Train the Model (Python)

### Training Script

```python
# scripts/train-model.py
import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Load your data ──
# X: feature matrix (n_samples, n_features)
# y: target matrix (n_samples, n_targets)
X, y = load_your_data()  # implement this

# ── Normalize ──
feature_scaler = StandardScaler().fit(X)
target_scaler = StandardScaler().fit(y)

X_norm = feature_scaler.transform(X)
y_norm = target_scaler.transform(y)

# ── Train ──
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),  # 2 hidden layers
    activation='relu',
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
)
model.fit(X_norm, y_norm)

# ── Export weights ──
weights = {
    "featureNames": FEATURE_NAMES,       # list of feature name strings
    "targetNames": TARGET_NAMES,         # list of target name strings
    "targetLabels": TARGET_LABELS,       # human-readable output labels
    "featureNorm": {
        "mean": feature_scaler.mean_.tolist(),
        "std": feature_scaler.scale_.tolist(),
    },
    "targetNorm": {
        "mean": target_scaler.mean_.tolist(),
        "std": target_scaler.scale_.tolist(),
    },
    "layers": [],
}

for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    weights["layers"].append({
        "weights": W.tolist(),   # shape: [in_size, out_size]
        "biases": b.tolist(),    # shape: [out_size]
    })

with open("model-weights.json", "w") as f:
    json.dump(weights, f)

print(f"Exported {len(weights['layers'])} layers")
print(f"Input: {len(FEATURE_NAMES)} features → Output: {len(TARGET_NAMES)} targets")
print(f"R² score: {model.score(X_norm, y_norm):.4f}")
```

### Key Training Decisions

**Feature engineering matters more than model complexity:**
- Derive domain-specific features from raw data (ratios, trends, aggregates)
- Use recency weighting if data is temporal (exponential decay)
- Normalize session counts by observation window (a user with 4 sessions over 2 weeks trains at the same rate as 16 sessions over 8 weeks)

**Imputation strategy:**
- Design imputation defaults during training so the inference code can handle missing features gracefully
- Use domain knowledge for defaults (e.g., interval pace ≈ tempo pace - 30 sec/km)
- Add a `data_completeness` feature (fraction of features that are non-null) so the model can self-calibrate confidence

**Sanity clamping:**
- After prediction, clamp outputs to physically possible ranges
- This prevents wild extrapolation on out-of-distribution inputs

## Step 2: Export Weights

The JSON weights file contains everything needed for inference:

```typescript
interface ModelWeights {
  featureNames: string[];              // what each input feature means
  targetNames: string[];               // what each output represents
  targetLabels: string[];              // human-readable output labels
  featureNorm: {
    mean: number[];                    // per-feature mean (from StandardScaler)
    std: number[];                     // per-feature std
  };
  targetNorm: {
    mean: number[];                    // per-target mean
    std: number[];                     // per-target std
  };
  layers: Array<{
    weights: number[][];               // [in_size, out_size]
    biases: number[];                  // [out_size]
  }>;
}
```

Commit this JSON file to your repo. It's typically 10-50 KB for a small MLP — negligible.

## Step 3: TypeScript Inference (Zero Dependencies)

### Forward Pass

```typescript
// lib/model-inference.ts

// eslint-disable-next-line @typescript-eslint/no-require-imports
const modelWeights = require('./model-weights.json');

interface ModelWeights {
  featureNames: string[];
  targetNames: string[];
  targetLabels: string[];
  featureNorm: { mean: number[]; std: number[] };
  targetNorm: { mean: number[]; std: number[] };
  layers: Array<{ weights: number[][]; biases: number[] }>;
}

function relu(x: number): number {
  return Math.max(0, x);
}

/** Matrix-vector multiply: W[in_size][out_size] × x[in_size] → result[out_size] */
function matVec(W: number[][], x: number[]): number[] {
  const out = new Array(W[0].length).fill(0);
  for (let j = 0; j < W[0].length; j++) {
    for (let i = 0; i < x.length; i++) {
      out[j] += W[i][j] * x[i];
    }
  }
  return out;
}

export function forwardPass(features: number[], weights: ModelWeights): number[] {
  const { featureNorm, targetNorm, layers } = weights;

  // 1. Normalize input
  let x = features.map((v, i) =>
    (v - featureNorm.mean[i]) / featureNorm.std[i]
  );

  // 2. Hidden layers with ReLU activation
  for (let l = 0; l < layers.length - 1; l++) {
    const { weights: W, biases: b } = layers[l];
    x = matVec(W, x).map((v, j) => relu(v + b[j]));
  }

  // 3. Output layer (no activation — linear)
  const lastLayer = layers[layers.length - 1];
  x = matVec(lastLayer.weights, x).map((v, j) => v + lastLayer.biases[j]);

  // 4. Denormalize output
  return x.map((v, i) => v * targetNorm.std[i] + targetNorm.mean[i]);
}
```

### Feature Extraction

```typescript
// lib/feature-extractor.ts

export interface Features {
  [key: string]: number | null;
}

/**
 * Extract features from your raw data.
 * This is the most important part — it must match your training pipeline exactly.
 */
export function extractFeatures(rawData: YourDataType[]): Features {
  // Filter to relevant time window
  const recent = filterToWindow(rawData, 8 * 7); // e.g., 8 weeks

  // Compute features that match your training featureNames
  return {
    feature_1: computeFeature1(recent),
    feature_2: computeFeature2(recent),
    // ... must match featureNames order exactly
  };
}

/**
 * Impute nulls with sensible defaults.
 * Returns a number[] ready for forwardPass().
 */
export function imputeFeatures(
  features: Features,
  featureNames: string[]
): number[] {
  const DEFAULTS: Record<string, number> = {
    feature_1: 0,
    feature_2: 0.5,
    // Derive defaults from available features when possible:
    // e.g., if feature_1 is missing but feature_2 exists,
    // estimate feature_1 from feature_2 using domain knowledge
  };

  return featureNames.map(name => features[name] ?? DEFAULTS[name] ?? 0);
}
```

### Recency Weighting (for temporal data)

```typescript
/** Exponential decay — recent observations matter more. */
const HALF_LIFE_DAYS = 28;
const LAMBDA = Math.log(2) / HALF_LIFE_DAYS;

function recencyWeight(dateStr: string, asOfDate: Date): number {
  const daysAgo = (asOfDate.getTime() - new Date(dateStr).getTime()) / 86400000;
  return Math.exp(-LAMBDA * Math.max(0, daysAgo));
}

/** Recency-weighted average. Returns null if no valid items. */
function weightedAvg(
  items: Array<{ value: number | null; date: string }>,
  asOf: Date
): number | null {
  const valid = items.filter(i => i.value !== null) as Array<{ value: number; date: string }>;
  if (valid.length === 0) return null;
  const weighted = valid.map(i => ({
    value: i.value,
    weight: recencyWeight(i.date, asOf),
  }));
  const totalW = weighted.reduce((s, i) => s + i.weight, 0);
  return weighted.reduce((s, i) => s + i.value * i.weight, 0) / totalW;
}
```

### Putting It Together

```typescript
// lib/predict.ts
import { forwardPass } from './model-inference';
import { extractFeatures, imputeFeatures } from './feature-extractor';

const weights = require('./model-weights.json');

export interface Prediction {
  values: Record<string, number>;
  hasData: boolean;
  confidence: number; // 0-1 based on data completeness
}

export function predict(rawData: YourDataType[]): Prediction {
  const features = extractFeatures(rawData);

  // Check minimum data requirements
  const requiredFeatures = ['feature_1', 'feature_2'];
  const missing = requiredFeatures.filter(k => features[k] === null);
  if (missing.length > requiredFeatures.length * 0.5) {
    return { values: {}, hasData: false, confidence: 0 };
  }

  // Compute data completeness (fed to model as a feature)
  const nullableKeys = Object.keys(features);
  const presentCount = nullableKeys.filter(k => features[k] !== null).length;
  const dataCompleteness = presentCount / nullableKeys.length;

  // Impute and predict
  const featureVec = imputeFeatures(features, weights.featureNames);
  const raw = forwardPass(featureVec, weights);

  // Sanity clamp outputs to physically possible ranges
  const clamped = raw.map((v, i) => {
    const mins = [/* your minimum valid values */];
    const maxs = [/* your maximum valid values */];
    return Math.max(mins[i], Math.min(maxs[i], v));
  });

  // Map to named outputs
  const values: Record<string, number> = {};
  weights.targetNames.forEach((name: string, i: number) => {
    values[name] = clamped[i];
  });

  return { values, hasData: true, confidence: dataCompleteness };
}
```

## Fitness & Sports Use Cases

This pattern is particularly powerful for fitness/sports apps where workout data is already being logged. The same architecture handles multiple prediction tasks — train different models on different targets, share the same TypeScript inference engine.

### Race Time Prediction
Predict 5K, 10K, Half Marathon, and Marathon times from training data.

**Features:** interval pace (recency-weighted), interval rep distance, HR ratio (avg/max), tempo pace, tempo distance, long run distance, long run pace, long run HR ratio, weekly running volume, session count (normalized to 8-week rate), pace trend (slope over time), data completeness.

**Targets:** race finish times in seconds (5K, 10K, HM, Marathon).

**Training data:** can be generated synthetically using the Riegel formula with noise, or collected from athletes with known race results + training logs.

### VO2 Max Estimation
Estimate VO2 max (ml/kg/min) without a lab test.

**Features:** best interval pace at HR ≥ 90% max, easy run pace at HR ~70% max, HR recovery rate (how fast HR drops after hard effort), long run HR drift (HR increase over constant-pace run), weekly volume, training age.

**Targets:** VO2 max (can be calibrated against Cooper test, VDOT tables, or actual lab data).

**Key insight:** the ratio of pace to HR at different intensities is a strong proxy for aerobic capacity. An athlete running 5:00/km at 155 bpm is fitter than one running the same pace at 175 bpm.

### Training Zone Prediction
Predict personalized lactate threshold, aerobic threshold, and training zones from workout data — without formal threshold testing.

**Features:** tempo pace + HR, easy pace + HR, interval pace + HR, pace-HR coupling over time (does the same pace require higher or lower HR as training progresses?), long run HR drift percentage.

**Targets:** threshold pace (sec/km), aerobic threshold pace (sec/km), max HR.

**Key insight:** the inflection point where HR rises disproportionately to pace increase approximates lactate threshold. With enough logged sessions at varying intensities, the model learns each athlete's individual threshold.

### Recovery & Readiness
Predict whether an athlete is ready for a hard session or needs more recovery.

**Features:** acute training load (last 7 days — total volume × intensity), chronic training load (last 28 days), ACWR (acute:chronic workload ratio), days since last rest, consecutive hard session count, sleep quality (if tracked), resting HR trend, RPE trend (is perceived effort increasing at same loads?).

**Targets:** readiness score (0-100), or binary classification (ready / needs recovery).

**Key insight:** ACWR between 0.8–1.3 is the "sweet spot" — below means detraining, above means injury risk. The model can learn individual athlete tolerances.

### Injury Risk Scoring
Flag when an athlete's training pattern suggests elevated injury risk.

**Features:** volume ramp rate (% increase week-over-week), training monotony (how repetitive — same sessions every week), training strain (monotony × load), consecutive days without rest, sudden intensity jumps, muscle group imbalance (ratio of lower to upper volume), recent complaints in workout notes (NLP features).

**Targets:** injury risk score, or binary within-30-days injury prediction (requires historical injury data for training).

### Strength Progression
Predict next-session working weight or estimated 1RM trajectory.

**Features:** per-lift history (weight × reps × sets over last 8 weeks), RPE trend, session frequency, training phase (hypertrophy vs. strength vs. peaking), body weight trend, concurrent training load (running volume competes for recovery).

**Targets:** predicted 1RM per lift, or recommended working weight for next session.

**Key insight:** the Epley formula (1RM = weight × (1 + reps/30)) gives a noisy but useful signal. An ML model trained on multiple sessions smooths out day-to-day variation and accounts for fatigue/recovery patterns that Epley alone misses.

### Implementation Pattern for Multiple Models

All fitness models share the same TypeScript inference engine — only the weights file and feature extractor differ:

```typescript
// lib/predictions.ts
import { forwardPass } from './model-inference';
import { extractRunFeatures } from './features/running';
import { extractStrengthFeatures } from './features/strength';
import { extractRecoveryFeatures } from './features/recovery';

const raceWeights = require('./weights/race-predictor.json');
const vo2Weights = require('./weights/vo2-estimator.json');
const recoveryWeights = require('./weights/recovery-scorer.json');

export function predictRaceTimes(workouts: Workout[]) {
  const features = extractRunFeatures(workouts);
  return forwardPass(imputeRun(features), raceWeights);
}

export function estimateVO2Max(workouts: Workout[]) {
  const features = extractRunFeatures(workouts); // same features, different model
  return forwardPass(imputeRun(features), vo2Weights);
}

export function recoveryScore(workouts: Workout[]) {
  const features = extractRecoveryFeatures(workouts);
  return forwardPass(imputeRecovery(features), recoveryWeights);
}
```

Each model is a separate JSON file (10-50 KB). The `forwardPass` function is shared. Feature extractors are grouped by domain. Total bundle size for 3-5 models: ~100-250 KB of weights + ~200 lines of TypeScript.

## Supported Model Types

### MLP (Multi-Layer Perceptron) — shown above
Best for: continuous predictions, regression, multi-output

### Linear Regression / Ridge / Lasso
Simplest case — single layer, no activation:

```typescript
// Single layer: y = W·x + b (then denormalize)
function linearPredict(features: number[], weights: ModelWeights): number[] {
  const { featureNorm, targetNorm, layers } = weights;
  let x = features.map((v, i) => (v - featureNorm.mean[i]) / featureNorm.std[i]);
  const { weights: W, biases: b } = layers[0];
  const out = matVec(W, x).map((v, j) => v + b[j]);
  return out.map((v, i) => v * targetNorm.std[i] + targetNorm.mean[i]);
}
```

### Random Forest / Gradient Boosting
Export as decision tree array, implement tree traversal:

```python
# Export script for Random Forest
def export_tree(tree, feature_names):
    t = tree.tree_
    return {
        "feature": [feature_names[i] if i >= 0 else None for i in t.feature],
        "threshold": t.threshold.tolist(),
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "value": [v[0].tolist() for v in t.value],
    }

weights = {
    "trees": [export_tree(t, FEATURE_NAMES) for t in model.estimators_],
    "featureNames": FEATURE_NAMES,
}
```

```typescript
// TypeScript tree traversal
function traverseTree(tree: ExportedTree, features: number[]): number[] {
  let node = 0;
  while (tree.feature[node] !== null) {
    const featureIdx = FEATURE_NAMES.indexOf(tree.feature[node]!);
    if (features[featureIdx] <= tree.threshold[node]) {
      node = tree.left[node];
    } else {
      node = tree.right[node];
    }
  }
  return tree.value[node];
}

function forestPredict(features: number[], weights: ForestWeights): number[] {
  const predictions = weights.trees.map(t => traverseTree(t, features));
  // Average across trees
  return predictions[0].map((_, i) =>
    predictions.reduce((sum, p) => sum + p[i], 0) / predictions.length
  );
}
```

## Best Practices

1. **Feature parity is critical** — the TypeScript feature extraction must produce the exact same features in the exact same order as the Python training pipeline. Test this with known inputs.

2. **Version your weights** — commit `model-weights.json` to git. When you retrain, the diff shows exactly what changed.

3. **Keep models small** — a 2-layer MLP with (64, 32) hidden units is ~10 KB of weights and runs in <1ms. You don't need a large model for most tabular prediction tasks.

4. **Sanity clamp outputs** — always clamp predictions to physically possible ranges. Neural networks can extrapolate wildly on out-of-distribution inputs.

5. **Add a data completeness feature** — let the model know how much of the input is real vs. imputed. This lets it self-calibrate.

6. **Use recency weighting for temporal data** — exponential decay with a 28-day half-life works well for fitness/health/activity data.

7. **Test with edge cases** — new users (minimal data), power users (lots of data), users with missing feature categories.

## When NOT to Use This Pattern

- **Image/audio/video models** — use ONNX runtime or TensorFlow.js instead
- **Models > 1MB of weights** — consider a server-side API
- **Models that need GPU** — this runs on CPU only
- **Frequently retrained models** — if you retrain daily, consider a model serving API
- **Deep learning (>3 layers)** — the hand-rolled forward pass works but gets unwieldy

## References

- [sklearn MLPRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- [StandardScaler docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- Pattern inspired by [Hypla](https://hypla.fit) race time predictor
