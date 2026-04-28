"""
device/config_qA.py
configuration for qA analysis.
"""

from __future__ import annotations
import floquet as ft

from landau_zener.analysis.detection import BranchHybridizationDetectConfig, ReferenceHybridizationConfig
from landau_zener.analysis.pipeline import SurveyConfig, ZoomConfig, RelevanceFilterConfig
from landau_zener.analysis.detuning_fit import LinearDetuningFitConfig
from landau_zener.analysis.metrics import IsolationConfig, CrowdingConfig
from landau_zener.analysis.slope_target import SlopeConfig
from landau_zener.utils.units import ghz_to_angular
QUBIT_NAME = "qA"
QUBIT_PARAMS = dict(
    EJ=4.1184,
    EC=0.8283,
    EL=1.1929,
    flux=0.501,
    cutoff=200,
)

OPTIONS = ft.Options(
    fit_range_fraction=0.6,
    floquet_sampling_time_fraction=0.0,
    fit_cutoff=4,
    overlap_cutoff=0.8,
    nsteps=30_000,
    num_cpus=1,
    save_floquet_modes=False,
)

SURVEY_CFG = SurveyConfig(
    nchi_survey=300,
    restrict_branches=None,
)

DETECT_CFG = BranchHybridizationDetectConfig(
    gap_max_MHz=650.0,
    g2_fit_half_window=20,
    observable="nbar",
    obs_edge_factor=4.0,
    min_edge_offset_pts=12,
    obs_swap_min_diff=0.5,
    min_separation_pts=10,
)

REF_CFG = ReferenceHybridizationConfig(
    ref_levels=(0, 1, 2),
    min_carrier_weight=0.15,
    hysteresis=0.0,
    gap_search_half_window_pts=40,
    g2_fit_half_window=20,
    partner_edge_factor=4.0,
    partner_min_edge_offset_pts=12,
)

ISO_CFG = IsolationConfig(iso_window_factor=3.0, iso_ratio_thresh=0.0)
CROWD_CFG = CrowdingConfig(crowd_window_factor=2.0)
ZOOM_CFG = ZoomConfig(
    enable_zoom=True,
    nchi_zoom=300,
    window_factor=8.0,
    max_zoom_events=50,
)
FILTER_CFG = RelevanceFilterConfig(
    P_ad_min=0.8,
    require_isolated_pair=True,
    iso_ratio_min=0.0,
    w_ref_min_over_set=0.0,
    max_events_keep=None,
)

LIN_CFG = LinearDetuningFitConfig(
    lin_scale_1=2.0,
    lin_scale_2=10.0,
    min_points=5,
    method="projection",
    fit_side="both",
)

SLOPE_CFG = SlopeConfig(
    chi_fit_max_MHz=20.0,
    nchi_fit=41,
    bare_a_levels=(0, 1, 2),
    bare_b_levels=range(12),
    photon_num=1,
    gap_max_MHz=650.0,
    window_factor=8.0,
    min_window_MHz=2.0,
    min_slope_diff=1e-4,
    max_candidates=200,
)

DEFAULT_DIM_Q = 25
DEFAULT_FLUX = 0.501
DEFAULT_CHI_RANGE_GHZ = (0.0, 0.2)

DEFAULT_DRIVE_GHZ = 4.56
DEFAULT_KAPPA_GHZ = 0.005


def kappa_res_angular() -> float:
    return ghz_to_angular(DEFAULT_KAPPA_GHZ)