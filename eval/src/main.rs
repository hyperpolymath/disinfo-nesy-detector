// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2024 Hyperpolymath

//! Evaluation pipeline CLI for Neuro-Symbolic Disinformation Detector
//!
//! Usage:
//!   eval-pipeline --dataset synthetic --seed 42
//!   eval-pipeline --dataset liar --path ./datasets/liar --split test

use anyhow::Result;
use clap::Parser;
use disinfo_eval::pipeline::{EvaluationConfig, EvaluationPipeline};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "eval-pipeline")]
#[command(about = "Evaluate disinformation detection models")]
#[command(version)]
struct Args {
    /// Dataset to evaluate on (synthetic, liar, isot)
    #[arg(short, long, default_value = "synthetic")]
    dataset: String,

    /// Path to dataset directory
    #[arg(short, long)]
    path: Option<PathBuf>,

    /// Random seed for reproducibility
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Evaluation split (train, validation, test)
    #[arg(long, default_value = "test")]
    split: String,

    /// Specific baselines to run (comma-separated, empty = all)
    #[arg(short, long)]
    baselines: Option<String>,

    /// Output directory for results
    #[arg(short, long, default_value = "eval/results")]
    output: PathBuf,

    /// Output format (json, markdown, both)
    #[arg(short, long, default_value = "both")]
    format: String,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    tracing::info!("Disinformation Detection Evaluation Pipeline");
    tracing::info!("============================================");
    tracing::info!("Dataset: {}", args.dataset);
    tracing::info!("Seed: {}", args.seed);
    tracing::info!("Split: {}", args.split);

    let baseline_names: Vec<String> = args
        .baselines
        .map(|b| b.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let config = EvaluationConfig {
        seed: args.seed,
        dataset_id: args.dataset.clone(),
        dataset_path: args.path.map(|p| p.to_string_lossy().to_string()),
        eval_split: args.split,
        run_baselines: true,
        baseline_names,
        output_dir: args.output.to_string_lossy().to_string(),
    };

    let mut pipeline = EvaluationPipeline::new(config);
    let results = pipeline.run()?;

    // Print summary to console
    println!("\n{}", "=".repeat(60));
    println!("EVALUATION SUMMARY");
    println!("{}", "=".repeat(60));
    println!("\nBest Model: {} (F1={:.4})", results.summary.best_model, results.summary.best_f1);
    println!("\nBaseline Comparison:");
    println!("{:-<60}", "");
    println!("{:<15} {:>10} {:>10} {:>10} {:>10}", "Model", "Accuracy", "F1", "MCC", "AUC-ROC");
    println!("{:-<60}", "");

    for baseline in &results.summary.baseline_comparison {
        let auc = baseline.auc_roc.map_or("-".to_string(), |v| format!("{:.4}", v));
        println!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            baseline.model, baseline.accuracy, baseline.f1_score, baseline.mcc, auc
        );
    }
    println!("{:-<60}", "");

    // Save outputs
    std::fs::create_dir_all(&args.output)?;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");

    if args.format == "json" || args.format == "both" {
        let json_path = args.output.join(format!("eval_{}_{}.json", args.dataset, timestamp));
        EvaluationPipeline::save_results(&results, &json_path)?;
        println!("\nJSON results saved to: {}", json_path.display());
    }

    if args.format == "markdown" || args.format == "both" {
        let report = EvaluationPipeline::generate_report(&results);
        let md_path = args.output.join(format!("eval_{}_{}.md", args.dataset, timestamp));
        std::fs::write(&md_path, report)?;
        println!("Markdown report saved to: {}", md_path.display());
    }

    println!("\nEvaluation complete!");

    Ok(())
}
