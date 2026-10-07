"""Unit tests for :mod:`plismbench.models`."""

from __future__ import annotations

from enum import Enum

from plismbench.models.bioptimus import H0Mini, HOptimus0, HOptimus1
from plismbench.models.cats import H0MiniGAP, SIPEH0Mini
from plismbench.models.extractor import Extractor
from plismbench.models.genbio_ai import GenBioPathFM
from plismbench.models.histai import HibouBase, HibouLarge
from plismbench.models.hkust import GPFM
from plismbench.models.kaiko_ai import KaikoViTBase, KaikoViTLarge, Midnight12k
from plismbench.models.lunit import LunitViTS8
from plismbench.models.mahmood_lab import CONCH, UNI, CONCHv15, UNI2h
from plismbench.models.meta import Dinov2ViTGiant
from plismbench.models.microsoft import ProvGigaPath
from plismbench.models.owkin import Phikon, PhikonV2
from plismbench.models.paige_ai import Virchow, Virchow2
from plismbench.models.standford import PLIP
from plismbench.models.vfmgeom import (
    H0MiniWithLeace,
    H0MiniWithPCAProj,
    HOptimus1WithLeace,
    UNI2hWithLeace,
)
from plismbench.models.vfmgeom_eraser import H0MiniWithEraser, HOptimus1WithEraser
from plismbench.models.vfmgeom_eraser_new import (
    H0MiniWithEraser as H0MiniWithEraserNew,
)
from plismbench.models.vfmgeom_eraser_new import (
    HOptimus1WithEraser as HOptimus1WithEraserNew,
)


class StringEnum(Enum):
    """A base class string enumerator."""

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def choices(cls):
        """Get Enum names."""
        return tuple(i.value for i in cls)


class FeatureExtractorsEnum(StringEnum):
    """A class enumerator for feature extractors."""

    # please follow the format "upper case = lower case"
    # this should map exactly the name in constants

    # Bioptimus
    H0_MINI = "h0_mini"
    HOPTIMUS0 = "hoptimus0"
    HOPTIMUS1 = "hoptimus1"  # access not granted for now
    # Kaiko AI
    KAIKO_VIT_BASE = "kaiko_vit_base"
    KAIKO_VIT_LARGE = "kaiko_vit_large"
    MIDNIGHT_12K = "midnight_12k"
    # Paige AI
    VIRCHOW = "virchow"
    VIRCHOW2 = "virchow2"
    # GenBio AI
    GENBIO_PATHFM = "genbio_pathfm"
    # Microsoft
    PROVGIGAPATH = "provgigapath"
    # Mahmood Lab
    CONCH = "conch"
    CONCHV15 = "conchv15"
    UNI = "uni"
    UNI2H = "uni2h"
    # HistAI
    HIBOU_BASE = "hibou_base"
    HIBOU_LARGE = "hibou_large"
    # Owkin
    PHIKON = "phikon"
    PHIKONV2 = "phikonv2"
    # HKUST
    GPFM = "gpfm"
    # Standford
    PLIP = "plip"
    # Lunit
    LUNIT_VIT_SMALL_8 = "lunit_vit_small_8"
    # Meta
    DINOV2_VIT_GIANT_IMAGENET = "dinov2_vit_giant_imagenet"
    # LEACE
    H0_MINI_WITH_LEACE = "h0_mini_with_leace"
    UNI2H_WITH_LEACE = "uni2h_with_leace"
    HOPTIMUS1_WITH_LEACE = "hoptimus1_with_leace"
    H0_MINI_GAP = "h0_mini_gap"
    SIPE_H0_MINI = "sipe_h0_mini"
    SIPE_CONCAT_H0_MINI = "sipe_concat_h0_mini"
    H0_MINI_WITH_PCA = "h0_mini_with_pca"
    H0_MINI_SCORPION_SLIDE_TO_MEAN_SOFT_LAM1000_FULLRANK = (
        "h0_mini_scorpion_slide_to_mean_soft_lam1000_fullrank"
    )
    H0_MINI_SCORPION_SLIDE_TO_MEAN_PCA_RANK64 = (
        "h0_mini_scorpion_slide_to_mean_pca_rank64"
    )
    HOPTIMUS1_SCORPION_SLIDE_TO_MEAN_SOFT_LAM10000_FULLRANK = (
        "hoptimus1_scorpion_slide_to_mean_soft_lam10000_fullrank"
    )
    HOPTIMUS1_SCORPION_SLIDE_TO_MEAN_SOFT_LAM1000_FULLRANK = (
        "hoptimus1_scorpion_slide_to_mean_soft_lam1000_fullrank"
    )
    HOPTIMUS1_SCORPION_SLIDE_TO_MEAN_PCA_RANK32 = (
        "hoptimus1_scorpion_slide_to_mean_pca_rank32"
    )

    HOPTIMUS1_SCORPION_MULTI_DELTAS_PCA_RANK128 = (
        "hoptimus1_scorpion_multi_deltas_pca_rank128"
    )

    HOPTIMUS1_SCORPION_MULTI_DELTAS_SOFT_LAM3000_RANKFULL = (
        "hoptimus1_scorpion_multi_deltas_soft_lam3000_rankfull"
    )

    HOPTIMUS1_SCORPION_CHAINED_DELTAS_PCA_RANK32_RANK64 = (
        "hoptimus1_scorpion_chained_deltas_pca_rank32_rank64"
    )
    HOPTIMUS1_SCORPION_MULTI_DELTAS_SOFT_LAM30_RANKFULL_MATCHXTRACE = (
        "hoptimus1_scorpion_multi_deltas_soft_lam30_rankfull_matchxtrace"
    )

    HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM30_LAM10_RANKFULL_MATCHXTRACE = (
        "hoptimus1_scorpion_chained_soft_lam30_lam10_rankfull_matchxtrace"
    )
    HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM30_LAM1_RANKFULL_MATCHXTRACE = (
        "hoptimus1_scorpion_chained_soft_lam30_lam1_rankfull_matchxtrace"
    )
    HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM10_LAM1_RANKFULL_MATCHXTRACE = (
        "hoptimus1_scorpion_chained_soft_lam10_lam1_rankfull_matchxtrace"
    )
    HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM10_LAM10_RANKFULL_MATCHXTRACE = (
        "hoptimus1_scorpion_chained_soft_lam10_lam10_rankfull_matchxtrace"
    )

    ### FINAL models

    H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM40 = (
        "h0_mini_scorpion_scanner_stage_soft_lam40"
    )
    H0_MINI_SCORPION_CHAINED_SOFT_LAM40_LAM1 = (
        "h0_mini_scorpion_chained_soft_lam40_lam1"
    )
    H0_MINI_SCORPION_SCANNER_STAGE_LEACE = "h0_mini_scorpion_scanner_stage_leace"
    H0_MINI_SCORPION_CHAINED_LEACE = "h0_mini_scorpion_chained_leace"
    H0_MINI_SCORPION_SCANNER_STAGE_PCA_RANK4 = (
        "h0_mini_scorpion_scanner_stage_pca_rank4"
    )
    H0_MINI_SCORPION_CHAINED_PCA_RANK4_RANK16 = (
        "h0_mini_scorpion_chained_pca_rank4_rank16"
    )
    ### FINAL models with sgd safe

    H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM70 = (
        "h0_mini_scorpion_scanner_stage_soft_lam70"
    )
    H0_MINI_SCORPION_CHAINED_SOFT_LAM70_LAM1 = (
        "h0_mini_scorpion_chained_soft_lam70_lam1"
    )
    H0_MINI_SCORPION_SCANNER_STAGE_PCA_RANK4_BIS = (
        "h0_mini_scorpion_scanner_stage_pca_rank4_bis"
    )
    H0_MINI_SCORPION_CHAINED_PCA_RANK4_RANK20 = (
        "h0_mini_scorpion_chained_pca_rank4_rank20"
    )
    ### FINAL models with sgd safe Hoptimus1
    HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM40 = (
        "hoptimus1_scorpion_scanner_stage_soft_lam40"
    )
    HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM40_LAM1 = (
        "hoptimus1_scorpion_chained_soft_lam40_lam1"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_LEACE = "hoptimus1_scorpion_scanner_stage_leace"
    HOPTIMUS1_SCORPION_CHAINED_LEACE = "hoptimus1_scorpion_chained_leace"
    HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK4 = (
        "hoptimus1_scorpion_scanner_stage_pca_rank4"
    )
    HOPTIMUS1_SCORPION_CHAINED_PCA_RANK4_RANK28 = (
        "hoptimus1_scorpion_chained_pca_rank4_rank28"
    )
    ### HARD erasers
    HOPTIMUS1_SCORPION_SCANNER_STAGE_HARD_FULL = (
        "hoptimus1_scorpion_scanner_stage_hard_full"
    )
    HOPTIMUS1_SCORPION_CHAINED_HARD_FULL_FULL = (
        "hoptimus1_scorpion_chained_hard_full_full"
    )
    ### FINAL Sweep
    HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM0P1 = (
        "hoptimus1_scorpion_scanner_stage_soft_lam0p1"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM1 = (
        "hoptimus1_scorpion_scanner_stage_soft_lam1"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM10 = (
        "hoptimus1_scorpion_scanner_stage_soft_lam10"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM100 = (
        "hoptimus1_scorpion_scanner_stage_soft_lam100"
    )

    ### FINAL Sweep h0 mini soft
    H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM0P1 = (
        "h0_mini_scorpion_scanner_stage_soft_lam0p1"
    )
    H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM1 = (
        "h0_mini_scorpion_scanner_stage_soft_lam1"
    )
    H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM10 = (
        "h0_mini_scorpion_scanner_stage_soft_lam10"
    )
    H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM100 = (
        "h0_mini_scorpion_scanner_stage_soft_lam100"
    )
    ### FINAL Sweep hoptimus1 PCA
    HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK2 = (
        "hoptimus1_scorpion_scanner_stage_pca_rank2"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK4_BIS = (
        "hoptimus1_scorpion_scanner_stage_pca_rank4_bis"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK8 = (
        "hoptimus1_scorpion_scanner_stage_pca_rank8"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK32 = (
        "hoptimus1_scorpion_scanner_stage_pca_rank32"
    )
    HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK128 = (
        "hoptimus1_scorpion_scanner_stage_pca_rank128"
    )


    def init(  # noqa: PLR0911, PLR0912
        self,
        device: int | list[int] | None,
        mixed_precision: bool = True,
        **kwargs,
    ) -> Extractor:
        """Initialize the feature extractor. Mixed precision is set by default."""
        if self is self.H0_MINI:
            return H0Mini(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HOPTIMUS0:
            return HOptimus0(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        # access not granted for now
        elif self is self.HOPTIMUS1:
            return HOptimus1(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.KAIKO_VIT_BASE:
            return KaikoViTBase(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.KAIKO_VIT_LARGE:
            return KaikoViTLarge(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.MIDNIGHT_12K:
            return Midnight12k(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.VIRCHOW:
            return Virchow(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.VIRCHOW2:
            return Virchow2(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.GENBIO_PATHFM:
            return GenBioPathFM(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PROVGIGAPATH:
            return ProvGigaPath(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.CONCH:
            return CONCH(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.CONCHV15:
            return CONCHv15(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.UNI:
            return UNI(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.UNI2H:
            return UNI2h(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HIBOU_BASE:
            return HibouBase(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HIBOU_LARGE:
            return HibouLarge(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PHIKON:
            return Phikon(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PHIKONV2:
            return PhikonV2(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.GPFM:
            return GPFM(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PLIP:
            return PLIP(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.DINOV2_VIT_GIANT_IMAGENET:
            return Dinov2ViTGiant(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.LUNIT_VIT_SMALL_8:
            return LunitViTS8(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.H0_MINI_WITH_LEACE:
            return H0MiniWithLeace(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.UNI2H_WITH_LEACE:
            return UNI2hWithLeace(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HOPTIMUS1_WITH_LEACE:
            return HOptimus1WithLeace(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.H0_MINI_GAP:
            return H0MiniGAP(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.SIPE_H0_MINI:
            return SIPEH0Mini(
                device=device,
                mixed_precision=mixed_precision,
                feature_mode="content",
                **kwargs,
            )
        elif self is self.SIPE_CONCAT_H0_MINI:
            return SIPEH0Mini(
                device=device,
                mixed_precision=mixed_precision,
                feature_mode="concat",
                **kwargs,
            )
        elif self is self.H0_MINI_WITH_PCA:
            return H0MiniWithPCAProj(
                device=device,
                mixed_precision=mixed_precision,
                same_dim=True,
                normalize_output=False,
                **kwargs,
            )

        elif self is self.H0_MINI_SCORPION_SLIDE_TO_MEAN_SOFT_LAM1000_FULLRANK:
            return H0MiniWithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/scorpion_slide_to_mean_soft_lam1000/eraser.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SLIDE_TO_MEAN_PCA_RANK64:
            return H0MiniWithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/scorpion_slide_to_mean_pca_rank64/eraser.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SLIDE_TO_MEAN_SOFT_LAM10000_FULLRANK:
            return HOptimus1WithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/h-optimus-1_scorpion_slide_to_mean_soft_lam10000/eraser.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SLIDE_TO_MEAN_SOFT_LAM1000_FULLRANK:
            return HOptimus1WithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/h-optimus-1_scorpion_slide_to_mean_soft_lam1000/eraser.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SLIDE_TO_MEAN_PCA_RANK32:
            return HOptimus1WithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/h-optimus-1_scorpion_slide_to_mean_pca_rank32/eraser.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_MULTI_DELTAS_PCA_RANK128:
            return HOptimus1WithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_multi_delta/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/scanner_plus_stain_paired_delta_pca_fold0_rank128_white1_shrinkA1_ridge0p0001.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_MULTI_DELTAS_SOFT_LAM3000_RANKFULL:
            return HOptimus1WithEraser(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_multi_delta_test/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/scanner_plus_stain_soft_delta_projection_fold0_full_lambda3000_shrinkA1_ridge0p0001.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_CHAINED_DELTAS_PCA_RANK32_RANK64:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_sequential_delta_test/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/fold0__combo3__scanner-r32-w1__stain_after_scanner-r64-w1.npz",
                **kwargs,
            )
        elif (
            self is self.HOPTIMUS1_SCORPION_MULTI_DELTAS_SOFT_LAM30_RANKFULL_MATCHXTRACE
        ):
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_multi_delta_weight_sweep/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/scanner_plus_stain_slide_w80_20_soft_delta_projection_fold0_full_lambda30_shrinkA1_ridge0p0001.npz",
                **kwargs,
            )
        elif (
            self
            is self.HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM30_LAM10_RANKFULL_MATCHXTRACE
        ):
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_sequential_delta_sweep/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/fold0__combo2__scanner-full-l30__stain_after_scanner-full-l10.npz",
                **kwargs,
            )
        elif (
            self is self.HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM30_LAM1_RANKFULL_MATCHXTRACE
        ):
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_sequential_delta_sweep/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/fold0__combo1__scanner-full-l30__stain_after_scanner-full-l1.npz",
                **kwargs,
            )
        elif (
            self is self.HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM10_LAM1_RANKFULL_MATCHXTRACE
        ):
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_sequential_delta_sweep_lambda_scanner_10/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/fold0__combo1__scanner-full-l10__stain_after_scanner-full-l1.npz",
                **kwargs,
            )
        elif (
            self
            is self.HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM10_LAM10_RANKFULL_MATCHXTRACE
        ):
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/experiments/SCORPION_tiles_224px_0p5mpp_sequential_delta_sweep_lambda_scanner_10/scorpion_scanner_stain_delta_grid/h-optimus-1_cls/fold_erasers/fold0__combo2__scanner-full-l10__stain_after_scanner-full-l10.npz",
                **kwargs,
            )
        ### FINAL for the paper
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM40:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner40_stain1_all_data/h0-mini_cls/components/00_scanner_soft_delta_projection_lambda40.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_CHAINED_SOFT_LAM40_LAM1:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner40_stain1_all_data/h0-mini_cls/chained_soft_scanner40_stain1_all_data.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_LEACE:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_LEACE/chained_leace_scanner_stain_all_data/h0-mini_cls/components/00_scanner_leace_svdtol0_01.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_CHAINED_LEACE:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_LEACE/chained_leace_scanner_stain_all_data/h0-mini_cls/chained_leace_scanner_stain_all_data.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_PCA_RANK4:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner4_stain16_all_data/h0-mini_cls/components/00_scanner_paired_delta_pca_rank4_white1.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_CHAINED_PCA_RANK4_RANK16:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner4_stain16_all_data/h0-mini_cls/chained_pca_scanner4_stain16_all_data.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM70:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner70_stain1_all_data/h0-mini_cls/components/00_scanner_soft_delta_projection_lambda70.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_CHAINED_SOFT_LAM70_LAM1:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner70_stain1_all_data/h0-mini_cls/chained_soft_scanner70_stain1_all_data.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_PCA_RANK4_BIS:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner4_stain20_all_data/h0-mini_cls/components/00_scanner_paired_delta_pca_rank4_white1.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_CHAINED_PCA_RANK4_RANK20:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner4_stain20_all_data/h0-mini_cls/chained_pca_scanner4_stain20_all_data.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM40:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner40_stain1_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_soft_delta_projection_lambda40.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_CHAINED_SOFT_LAM40_LAM1:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner40_stain1_all_data_hoptimus1/h-optimus-1_cls/chained_soft_scanner40_stain1_all_data_hoptimus1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_LEACE:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_LEACE/chained_leace_scanner_stain_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_leace_svdtol0_01.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_CHAINED_LEACE:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_LEACE/chained_leace_scanner_stain_all_data_hoptimus1/h-optimus-1_cls/chained_leace_scanner_stain_all_data_hoptimus1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK4:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_pca_hoptimus1/chained_pca_scanner4_stain28_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_paired_delta_pca_rank4_white1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_CHAINED_PCA_RANK4_RANK28:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_pca_hoptimus1/chained_pca_scanner4_stain28_all_data_hoptimus1/h-optimus-1_cls/chained_pca_scanner4_stain28_all_data_hoptimus1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_HARD_FULL:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_hard/chained_hard_scannerFull_stainFull_all_data/h-optimus-1_cls/components/00_scanner_hard_delta_projection_full.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_CHAINED_HARD_FULL_FULL:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers/SCORPION_tiles_224px_0p5mpp_sequential_hard/chained_hard_scannerFull_stainFull_all_data/h-optimus-1_cls/chained_hard_scannerFull_stainFull_all_data.npz",
                **kwargs,
            )
        ## FInal Sweep
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM0P1:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner0p1_stain1_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_soft_delta_projection_lambda0_1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM1:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner1_stain1_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_soft_delta_projection_lambda1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM10:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner10_stain1_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_soft_delta_projection_lambda10.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_SOFT_LAM100:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner100_stain1_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_soft_delta_projection_lambda100.npz",
                **kwargs,
            )
        ## FInal Sweep H0 mini soft
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM0P1:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner0p1_stain1_all_data_h0mini/h0-mini_cls/components/00_scanner_soft_delta_projection_lambda0_1.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM1:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner1_stain1_all_data_h0mini/h0-mini_cls/components/00_scanner_soft_delta_projection_lambda1.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM10:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner10_stain1_all_data_h0mini/h0-mini_cls/components/00_scanner_soft_delta_projection_lambda10.npz",
                **kwargs,
            )
        elif self is self.H0_MINI_SCORPION_SCANNER_STAGE_SOFT_LAM100:
            return H0MiniWithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_soft/chained_soft_scanner100_stain1_all_data_h0mini/h0-mini_cls/components/00_scanner_soft_delta_projection_lambda100.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK2:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner2_stain28_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_paired_delta_pca_rank2_white1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK4_BIS:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner4_stain28_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_paired_delta_pca_rank4_white1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK8:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner8_stain28_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_paired_delta_pca_rank8_white1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK32:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner32_stain28_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_paired_delta_pca_rank32_white1.npz",
                **kwargs,
            )
        elif self is self.HOPTIMUS1_SCORPION_SCANNER_STAGE_PCA_RANK128:
            return HOptimus1WithEraserNew(
                device=device,
                mixed_precision=mixed_precision,
                eraser_path="/home/valentin/workspaces/vfm-geom-xai/outputs/final_erasers_sweep/SCORPION_tiles_224px_0p5mpp_sequential_pca/chained_pca_scanner128_stain28_all_data_hoptimus1/h-optimus-1_cls/components/00_scanner_paired_delta_pca_rank128_white1.npz",
                **kwargs,
            )

        else:
            raise NotImplementedError(f"Extractor {self} is not supported.")
