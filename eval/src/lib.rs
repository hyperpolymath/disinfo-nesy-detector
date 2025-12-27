// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2024 Hyperpolymath

//! Evaluation pipeline for Neuro-Symbolic Disinformation Detector
//!
//! This crate provides:
//! - Dataset loading and preprocessing (LIAR, ISOT, FEVER, etc.)
//! - Evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
//! - Baseline models (Random, Majority, TF-IDF + Logistic Regression)
//! - Reproducible evaluation pipeline with seeded randomness

pub mod datasets;
pub mod metrics;
pub mod baselines;
pub mod pipeline;

pub use datasets::{Dataset, DatasetConfig, Sample, Label};
pub use metrics::{EvaluationMetrics, ConfusionMatrix, ClassificationReport};
pub use baselines::{BaselineModel, RandomBaseline, MajorityBaseline, TfIdfBaseline};
pub use pipeline::{EvaluationPipeline, EvaluationConfig, EvaluationResults};
