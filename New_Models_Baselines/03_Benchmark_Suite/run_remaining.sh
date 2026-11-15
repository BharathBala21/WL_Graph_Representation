#!/usr/bin/env bash
# Runs the remaining experiments one after another (so timings are not disturbed), from this folder.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CNN="$HERE/../04_CNN_Frozen_Features"
cd "$HERE"
echo "== level3 (feature models)";            python run_level3.py --models all
cd "$CNN"
echo "== cnn cache";                          python cnn_pipeline.py cache
echo "== cnn robust-cache";                   python cnn_pipeline.py robust-cache
echo "== cnn embed";                          python cnn_pipeline.py embed
echo "== cnn evaluate (level 1, level 2)";    python cnn_pipeline.py evaluate
echo "== cnn level3";                         python cnn_pipeline.py level3
echo "== cnn learning curves";                python cnn_pipeline.py curves
echo "== cnn robustness";                     python cnn_pipeline.py robust
cd "$HERE"
echo "== learning curves (round-2 models)";   python run_learning_curves.py level2 level1 --models round2
echo "== robustness (all feature models)";    python run_robustness.py evaluate
cd "$CNN"
echo "== cnn robustness (re-merge after feature models)"; python cnn_pipeline.py robust
cd "$HERE"
echo "== report";                             python make_report.py
echo "ALL DONE"
