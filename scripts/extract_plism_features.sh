#!/usr/bin/env bash
set -euo pipefail

extractors=(
    # h0_mini_scorpion_scanner_stage_soft_lam40
    # h0_mini_scorpion_chained_soft_lam40_lam1
    # h0_mini_scorpion_scanner_stage_leace
    # h0_mini_scorpion_chained_leace
    # h0_mini_scorpion_scanner_stage_pca_rank4
    # h0_mini_scorpion_chained_pca_rank4_rank16
    # h0_mini_scorpion_scanner_stage_soft_lam70
    # h0_mini_scorpion_chained_soft_lam70_lam1
    # h0_mini_scorpion_scanner_stage_pca_rank4_bis
    # h0_mini_scorpion_chained_pca_rank4_rank20
    # h0_mini_scorpion_scanner_stage_leace
    # h0_mini_scorpion_chained_leace
    # hoptimus1_scorpion_scanner_stage_soft_lam40
    # hoptimus1_scorpion_chained_soft_lam40_lam1
    # hoptimus1_scorpion_scanner_stage_leace
    # hoptimus1_scorpion_chained_leace
    # hoptimus1_scorpion_scanner_stage_pca_rank4
    # hoptimus1_scorpion_chained_pca_rank4_rank28
    # hoptimus1_scorpion_scanner_stage_hard_full
    # hoptimus1_scorpion_chained_hard_full_full

    # Extraction for the sweep analysis
    # hoptimus1_scorpion_scanner_stage_soft_lam0p1
    # hoptimus1_scorpion_scanner_stage_soft_lam1
    # hoptimus1_scorpion_scanner_stage_soft_lam10
    # hoptimus1_scorpion_scanner_stage_soft_lam100

    # h0_mini_scorpion_scanner_stage_soft_lam0p1
    # h0_mini_scorpion_scanner_stage_soft_lam1
    # h0_mini_scorpion_scanner_stage_soft_lam10
    # h0_mini_scorpion_scanner_stage_soft_lam100
    
    hoptimus1_scorpion_scanner_stage_pca_rank2
    hoptimus1_scorpion_scanner_stage_pca_rank4_bis
    hoptimus1_scorpion_scanner_stage_pca_rank8
    hoptimus1_scorpion_scanner_stage_pca_rank32
    hoptimus1_scorpion_scanner_stage_pca_rank128

)

PLISM_ENV="plismbench"
PLISM_BATCH_SIZE=128
PLISM_FEATURES_DIR="/home/valentin/workspaces/plism-benchmark/output/features"
PLISM_DOWNLOAD_DIR="/mnt/nas7/data/Personal/Valentin/plism"
PLISM_NUM_WORKERS=8

for extractor in "${extractors[@]}"; do
    echo "Running PLISM extraction with extractor: ${extractor}"

    conda run --no-capture-output -n "${PLISM_ENV}" \
        plismbench extract \
        --extractor "${extractor}" \
        --batch-size "${PLISM_BATCH_SIZE}" \
        --export-dir "${PLISM_FEATURES_DIR}" \
        --download-dir "${PLISM_DOWNLOAD_DIR}" \
        --workers "${PLISM_NUM_WORKERS}"
done