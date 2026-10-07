# List of feature extractors to compute metrics on
extractors=(
    # h0_mini
    # h0_mini_gap
    # sipe_h0_mini
    # sipe_concat_h0_mini
    # h0_mini_with_leace
    # h0_mini_with_pca
    # uni2h_with_leace
    # uni2h
    # hoptimus1_with_leace
    # hoptimus1
    # h0_mini_scorpion_slide_to_mean_soft_lam1000_fullrank
    # h0_mini_scorpion_slide_to_mean_pca_rank64
    # hoptimus1_scorpion_slide_to_mean_soft_lam10000_fullrank
    # hoptimus1_scorpion_slide_to_mean_soft_lam1000_fullrank
    # hoptimus1_scorpion_slide_to_mean_pca_rank32
    # hoptimus1_scorpion_multi_deltas_pca_rank128
    # hoptimus1_scorpion_multi_deltas_soft_lam3000_rankfull
    # hoptimus1_scorpion_chained_deltas_pca_rank32_rank64
    # hoptimus1_scorpion_multi_deltas_soft_lam30_rankfull_matchxtrace
    # hoptimus1_scorpion_chained_soft_lam30_lam10_rankfull_matchxtrace
    # hoptimus1_scorpion_chained_soft_lam30_lam1_rankfull_matchxtrace
    # hoptimus1_scorpion_chained_soft_lam10_lam1_rankfull_matchxtrace
    # hoptimus1_scorpion_chained_soft_lam10_lam10_rankfull_matchxtrace
    # # Final
    # h0_mini_scorpion_scanner_stage_soft_lam40
    # h0_mini_scorpion_chained_soft_lam40_lam1
    # h0_mini_scorpion_scanner_stage_leace
    # h0_mini_scorpion_chained_leace
    # h0_mini_scorpion_scanner_stage_pca_rank4
    # h0_mini_scorpion_chained_pca_rank4_rank16
    # # Final sgd safe
    # h0_mini_scorpion_scanner_stage_soft_lam70
    # h0_mini_scorpion_chained_soft_lam70_lam1
    # h0_mini_scorpion_scanner_stage_pca_rank4_bis
    # h0_mini_scorpion_chained_pca_rank4_rank20
    # # Final sgd safe hoptimus1
    # hoptimus1_scorpion_scanner_stage_soft_lam40
    # hoptimus1_scorpion_chained_soft_lam40_lam1
    # hoptimus1_scorpion_scanner_stage_leace
    # hoptimus1_scorpion_chained_leace
    # hoptimus1_scorpion_scanner_stage_pca_rank4
    # hoptimus1_scorpion_chained_pca_rank4_rank28
    # Hard erasers
    # hoptimus1_scorpion_scanner_stage_hard_full
    # hoptimus1_scorpion_chained_hard_full_full

    # Mean Oracle
    # h0_mini_oracle_mean
    # hoptimus1_oracle_mean
    # uni2h_oracle_mean

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

# List of number of tiles to compute metrics on: the reference for leaderboard is 8,139 !
n_tiles=(
    8139
)

# Set features and metrics directories
features_dir=/home/valentin/workspaces/plism-benchmark/output/features/
metrics_dir=/home/valentin/workspaces/plism-benchmark/output/metrics/

# Iterate over number of tiles
for _n_tiles in ${n_tiles[*]}
do
    # Iterate over extractors
    for extractor in ${extractors[*]}
    do
        plismbench evaluate \
        --extractor "${extractor}" \
        --features-dir $features_dir \
        --metrics-dir $metrics_dir \
        --n-tiles $_n_tiles \
        --device "gpu"
    done
done
