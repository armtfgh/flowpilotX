"""FLORA-Design — Parameter Agent: fills numerical parameters using RAG."""

import logging
import math

from flora_design.rules.unit_op_rules import PARAMETER_DEFAULTS
from flora_translate.schemas import ChemFeatures, ProcessTopology

logger = logging.getLogger("flora.design.parameters")


class ParameterAgent:
    """Fill and validate numerical parameters in the topology."""

    def run(
        self,
        topology: ProcessTopology,
        features: ChemFeatures,
        records: dict,
    ) -> ProcessTopology:
        """Fill missing parameters and run consistency checks."""
        defaults = PARAMETER_DEFAULTS.get(
            features.reaction_class, PARAMETER_DEFAULTS["unknown"]
        )

        # Extract literature values if available
        lit_values = self._extract_literature_values(records)

        # Fill topology-level parameters if missing
        if not topology.residence_time_min:
            topology.residence_time_min = (
                lit_values.get("residence_time_min")
                or defaults["residence_time_min"]
            )
        if not topology.total_flow_rate_mL_min:
            topology.total_flow_rate_mL_min = (
                lit_values.get("flow_rate_mL_min")
                or defaults["flow_rate_mL_min"]
            )

        # Compute reactor volume for consistency
        topology.reactor_volume_mL = round(
            topology.residence_time_min * topology.total_flow_rate_mL_min, 2
        )

        # Fill unit operation parameters
        for op in topology.unit_operations:
            if op.op_type == "coil_reactor":
                p = op.parameters
                if not p.get("ID_mm"):
                    p["ID_mm"] = lit_values.get("tubing_ID_mm") or defaults["tubing_ID_mm"]
                p["volume_mL"] = topology.reactor_volume_mL
                p["temperature_C"] = (
                    features.temperature_C
                    or lit_values.get("temperature_C")
                    or defaults["temperature_C"]
                )
                # Compute tubing length
                id_m = p["ID_mm"] * 1e-3
                area = math.pi * (id_m / 2) ** 2
                vol_m3 = topology.reactor_volume_mL * 1e-6
                p["length_m"] = round(vol_m3 / area, 2) if area > 0 else 0

            elif op.op_type == "bpr":
                if not op.parameters.get("pressure_bar"):
                    op.parameters["pressure_bar"] = (
                        lit_values.get("BPR_bar") or defaults["BPR_bar"]
                    )

            elif op.op_type == "led_module":
                if not op.parameters.get("wavelength_nm"):
                    op.parameters["wavelength_nm"] = features.wavelength_nm
                if not op.parameters.get("power_W"):
                    op.parameters["power_W"] = lit_values.get("power_W", 40)

            elif op.op_type == "pump":
                fr = topology.total_flow_rate_mL_min / max(
                    1, sum(1 for o in topology.unit_operations if o.op_type == "pump")
                )
                op.parameters["flow_rate_mL_min"] = round(fr, 3)

        # Consistency warnings
        self._check_consistency(topology)

        return topology

    def _extract_literature_values(self, records: dict) -> dict:
        """Extract median parameter values from retrieved records."""
        if not records or not records.get("metadatas") or not records["metadatas"][0]:
            return {}
        # ChromaDB metadata has limited fields; return what's available
        # In practice, full record data would be loaded for richer extraction
        return {}

    def _check_consistency(self, topology: ProcessTopology) -> None:
        """Log warnings for inconsistent parameters."""
        computed_vol = topology.residence_time_min * topology.total_flow_rate_mL_min
        if abs(computed_vol - topology.reactor_volume_mL) > 0.1:
            logger.warning(
                f"Volume inconsistency: tau*Q={computed_vol:.2f} != "
                f"V={topology.reactor_volume_mL:.2f}"
            )

        # Check tubing length
        for op in topology.unit_operations:
            if op.op_type == "coil_reactor":
                length = op.parameters.get("length_m", 0)
                if length > 30:
                    logger.warning(
                        f"Very long reactor ({length:.1f}m). Consider "
                        f"increasing flow rate or shorter residence time."
                    )
                elif length < 0.5:
                    logger.warning(
                        f"Very short reactor ({length:.1f}m). Consider "
                        f"reducing flow rate."
                    )
