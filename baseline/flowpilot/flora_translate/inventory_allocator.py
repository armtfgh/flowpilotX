"""Deterministic allocation of physical inventory to process operations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable

from flora_translate.inventory_constraints import (
    equipment_system_compatible,
    selected_process_systems,
)
from flora_translate.schemas import FlowProposal, LabInventory, ProcessTopology
from flora_translate.pump_settings import setting_supported


AVAILABLE_STATUSES = {"available", "ready", "in_service", "in service"}
REACTOR_TYPES = {
    "coil_reactor", "photoreactor", "reactor", "heated_coil",
    "packed_bed", "packed_bed_reactor", "chip_reactor", "microreactor",
    "microchannel", "chip", "microfluidic",
}
MIXER_TYPES = {"mixer", "t_mixer", "y_mixer", "quench_mixer"}


class InventoryAllocator:
    """Compile abstract unit operations into inventory-backed operations."""

    def __init__(self, inventory: LabInventory, proposal: FlowProposal):
        self.inventory = inventory
        self.proposal = proposal
        self.strict = bool(inventory.strict_assignment)
        self.remaining = {
            item.equipment_id: item.quantity for item in inventory.all_equipment()
        }
        self.assignments: list[dict[str, Any]] = []
        self.unresolved: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.assumed_accessories: list[dict[str, Any]] = []
        self.used_reactor_ids: list[str] = []
        self.selected_systems = selected_process_systems(proposal, inventory)
        self.compiled_operations: list[Any] = []
        self.resource_remaining = dict(inventory.resource_capacities)

    def compile(self, topology: ProcessTopology) -> tuple[ProcessTopology, dict[str, Any]]:
        compiled = deepcopy(topology)
        self._remove_redundant_mixers(compiled)
        self._insert_required_gas_accessories(compiled)
        self.compiled_operations = compiled.unit_operations
        incoming = _incoming_counts(compiled)

        for operation in compiled.unit_operations:
            self._allocate_operation(operation, incoming.get(operation.op_id, 0))

        self._attach_auxiliary_equipment(compiled)
        self._allocate_tubing()
        self._validate_reactor_train(compiled)
        manifest = self._instrument_manifest()
        complete = not self.unresolved
        has_assumptions = bool(self.assumed_accessories)
        compiled.compilation_status = (
            "inventory_assigned_with_accessories"
            if complete and has_assumptions
            else "inventory_assigned"
            if complete
            else "inventory_incomplete"
            if self.strict
            else "legacy_compatible"
        )
        compiled.inventory_schema_version = self.inventory.schema_version
        compiled.inventory_sha256 = _inventory_sha256(self.inventory)
        compiled.instrument_manifest = manifest
        compiled.pid_description = " -> ".join(
            operation.label
            for operation in compiled.unit_operations
            if operation.op_type != "led_module"
        )

        report = {
            "schema_version": "flowpilot_inventory_allocation_v1.0",
            "status": (
                "complete_with_assumptions"
                if complete and has_assumptions
                else "complete"
                if complete
                else "incomplete"
            ),
            "strict_assignment": self.strict,
            "inventory_schema_version": self.inventory.schema_version,
            "inventory_sha256": compiled.inventory_sha256,
            "checks": {
                "all_required_operations_assigned": complete,
                "equipment_quantities_respected": all(value >= 0 for value in self.remaining.values()),
                "no_unknown_equipment_ids": self._no_unknown_ids(),
                "serial_reactor_train_valid": not any(
                    item["requirement_id"] == "INV-REACTOR-TRAIN" for item in self.unresolved
                ),
                "standard_accessories_verified": not has_assumptions,
            },
            "assignments": self.assignments,
            "instrument_manifest": manifest,
            "unresolved_requirements": self.unresolved,
            "assumed_standard_accessories": self.assumed_accessories,
            "warnings": self.warnings,
        }
        return compiled, report

    def _allocate_operation(self, operation, input_count: int) -> None:
        op_type = operation.op_type.lower()
        if op_type == "pump":
            flow = _num(operation.parameters.get("flow_rate_mL_min"))
            requested_id = str(
                operation.parameters.get("inventory_equipment_id") or ""
            )
            stream_identity = " ".join(
                [
                    *(str(item) for item in operation.parameters.get("contents") or []),
                    str(operation.parameters.get("solvent") or ""),
                    str(operation.label or ""),
                ]
            )
            candidates = [
                item for item in self.inventory.pumps
                if _available(item)
                and (not requested_id or item.equipment_id == requested_id)
                and item.min_flow_rate_mL_min - 1e-12 <= flow <= item.max_flow_rate_mL_min + 1e-12
                and setting_supported(item, flow)
                and self.proposal.BPR_bar <= item.max_pressure_bar + 1e-12
                and _pump_material_compatible(item, stream_identity)
                and equipment_system_compatible(item, self.selected_systems)
            ]
            self._claim(
                operation,
                "pumps",
                candidates,
                {"equipment_id": requested_id or None, "flow_rate_mL_min": flow},
            )
            return
        if op_type == "mfc":
            from flora_translate.gas_settings import gas_setting_supported
            gas = _operation_gas(operation)
            flow = _num(operation.parameters.get("gas_flow_sccm"))
            requested_id = str(
                operation.parameters.get("inventory_equipment_id") or ""
            )
            candidates = [
                item for item in self.inventory.gas_hardware
                if _available(item)
                and "mfc" in item.type.lower()
                and (not requested_id or item.equipment_id == requested_id)
                and (not gas or not item.gas or _gas_matches(gas, item.gas))
                and (item.min_flow_sccm is None or flow >= item.min_flow_sccm - 1e-12)
                and (item.max_flow_sccm is None or flow <= item.max_flow_sccm + 1e-12)
                and gas_setting_supported(item, flow)
                and (item.max_pressure_bar is None or self.proposal.BPR_bar <= item.max_pressure_bar + 1e-12)
            ]
            self._claim(
                operation,
                "gas_hardware",
                candidates,
                {"equipment_id": requested_id or None, "gas": gas, "flow_sccm": flow},
            )
            return
        if op_type in MIXER_TYPES:
            total_flow = _num(operation.parameters.get("Q_inlet_mL_min"), self.proposal.flow_rate_mL_min)
            if not self.inventory.mixers:
                self._assume_standard_accessory(
                    operation,
                    "mixers",
                    {
                        "type": "T-mixer",
                        "input_count": max(input_count, 2),
                        "total_flow_mL_min": total_flow,
                        "pressure_bar": self.proposal.BPR_bar,
                    },
                )
                return
            candidates = [
                item for item in self.inventory.mixers
                if _available(item)
                and item.max_inputs >= max(input_count, 2)
                and (item.min_total_flow_mL_min is None or total_flow >= item.min_total_flow_mL_min - 1e-12)
                and (item.max_total_flow_mL_min is None or total_flow <= item.max_total_flow_mL_min + 1e-12)
                and (item.max_pressure_bar is None or self.proposal.BPR_bar <= item.max_pressure_bar + 1e-12)
                and equipment_system_compatible(item, self.selected_systems)
            ]
            self._claim(operation, "mixers", candidates, {"input_count": input_count, "total_flow_mL_min": total_flow})
            return
        if op_type in REACTOR_TYPES:
            self._allocate_reactor(operation)
            return
        if op_type == "led_module":
            from flora_translate.equipment_resources import light_fits_stage
            prefix = operation.op_id.rsplit("_", 1)[0]
            reactor_op = next((op for op in self.compiled_operations
                               if op.op_type.lower() in REACTOR_TYPES and op.op_id.startswith(prefix)), None)
            reactor_id = (reactor_op.parameters.get("inventory_equipment_id") if reactor_op else None)
            reactor = next((r for r in self.inventory.reactors if r.equipment_id == reactor_id), None)
            wavelength = _num(operation.parameters.get("wavelength_nm"), self.proposal.wavelength_nm)
            temperature = _num(
                operation.parameters.get("temperature_C"),
                self.proposal.temperature_C,
            )
            requested_id = str(
                operation.parameters.get("inventory_equipment_id") or ""
            )
            candidates = [
                item for item in self.inventory.light_sources
                if _available(item)
                and (not requested_id or item.equipment_id == requested_id)
                and abs(item.wavelength_nm - wavelength) <= max(5.0, wavelength * 0.02)
                and _light_temperature_supported(item, temperature)
                and (light_fits_stage(item, reactor, self.proposal, self.inventory)
                     if reactor is not None else not item.module_id)
                and equipment_system_compatible(item, self.selected_systems)
            ]
            self._claim(
                operation,
                "light_sources",
                candidates,
                {
                    "equipment_id": requested_id or None,
                    "wavelength_nm": wavelength,
                    "temperature_C": temperature,
                },
            )
            return
        if op_type == "check_valve":
            candidates = [item for item in self.inventory.safety_accessories
                          if _available(item) and item.type == "check_valve"
                          and (item.max_pressure_bar is None or self.proposal.BPR_bar <= item.max_pressure_bar)]
            self._claim(operation, "safety_accessories", candidates, {"direction": "toward reactor"})
            selected = next((x for x in candidates if x.equipment_id == operation.inventory_item_id), None)
            if selected:
                operation.parameters["cracking_pressure_bar"] = selected.cracking_pressure_bar
                if selected.max_pressure_bar is None:
                    self.warnings.append(f"{selected.name}: maximum pressure rating not supplied; verify before laboratory execution. Cracking pressure is not a pressure rating.")
            return
        if op_type == "bpr":
            pressure = _num(operation.parameters.get("pressure_bar"), self.proposal.BPR_bar)
            candidates = [
                item for item in self.inventory.pressure_controllers
                if _available(item) and _pressure_supported(item, pressure)
                and equipment_system_compatible(item, self.selected_systems)
            ]
            self._claim(operation, "pressure_controllers", candidates, {"pressure_bar": pressure})
            return
        if op_type in {"deoxygenation_unit", "degas", "degasser"}:
            flow = self.proposal.flow_rate_mL_min
            candidates = [
                item for item in self.inventory.degassers
                if _available(item)
                and (item.min_flow_mL_min is None or flow >= item.min_flow_mL_min - 1e-12)
                and (item.max_flow_mL_min is None or flow <= item.max_flow_mL_min + 1e-12)
                and (item.max_pressure_bar is None or self.proposal.BPR_bar <= item.max_pressure_bar + 1e-12)
            ]
            self._claim(operation, "degassers", candidates, {"flow_rate_mL_min": flow})
            return
        if op_type in {"inline_filter", "filter"}:
            candidates = [
                item for item in self.inventory.filters
                if _available(item)
                and (item.max_pressure_bar is None or self.proposal.BPR_bar <= item.max_pressure_bar + 1e-12)
            ]
            self._claim(operation, "filters", candidates, {})
            return
        if op_type in {"separator", "phase_separator", "liq_liq_extraction"}:
            candidates = [item for item in self.inventory.separators if _available(item)]
            self._claim(operation, "separators", candidates, {})
            return
        if op_type in {"heat_exchanger", "heater", "chiller"}:
            temperature = _num(
                operation.parameters.get("temperature_C"),
                self.proposal.temperature_C,
            )
            candidates = [
                item for item in self.inventory.temperature_controllers
                if _available(item) and _temperature_supported(item, temperature)
            ]
            self._claim(
                operation,
                "temperature_controllers",
                candidates,
                {"temperature_C": temperature},
            )
            return
        if op_type == "collector":
            if self.inventory.collectors:
                candidates = [item for item in self.inventory.collectors if _available(item)]
                self._claim(operation, "collectors", candidates, {})
            else:
                operation.assignment_status = "process_endpoint"
                operation.inventory_category = "process_endpoint"
            return

        if operation.required:
            self._unresolved(operation, "unsupported_equipment", f"No allocator exists for required operation type '{op_type}'.")

    def _allocate_reactor(self, operation) -> None:
        parameters = operation.parameters or {}
        volume = _num(parameters.get("volume_mL"), self.proposal.reactor_volume_mL)
        diameter = _num(parameters.get("ID_mm"), self.proposal.tubing_ID_mm)
        material = str(parameters.get("material") or self.proposal.tubing_material or "").lower()
        temperature = _num(parameters.get("temperature_C"), self.proposal.temperature_C)
        selected_id = str((self.proposal.inventory_selection or {}).get("equipment_id") or "")
        operation_selected_id = str(
            parameters.get("inventory_equipment_id") or ""
        )

        candidates = [
            item for item in self.inventory.reactors
            if _available(item)
            and (not operation_selected_id or item.equipment_id == operation_selected_id)
            and not item.component_volumes_mL
            and str(item.configuration or "").lower() not in {"serial", "series"}
            and abs(item.volume_mL - volume) <= max(0.02, 0.005 * max(volume, 1.0))
            and abs(item.ID_mm - diameter) <= 1e-3
            and (not material or item.material.lower() == material)
            and _reactor_temperature_supported(item, temperature)
            and (item.max_pressure_bar is None or self.proposal.BPR_bar <= item.max_pressure_bar + 1e-12)
            and equipment_system_compatible(item, self.selected_systems)
        ]
        candidates.sort(
            key=lambda item: (
                item.equipment_id != operation_selected_id
                if operation_selected_id
                else item.equipment_id != selected_id,
                item.name,
                item.equipment_id,
            )
        )
        if self._claim(
            operation,
            "reactors",
            candidates,
            {
                "equipment_id": operation_selected_id or None,
                "volume_mL": volume,
                "ID_mm": diameter,
                "temperature_C": temperature,
            },
            unresolved=False,
        ):
            self.used_reactor_ids.append(operation.inventory_item_id)
            return

        train = self._matching_train(volume)
        if train is not None and self._claim_train(operation, train):
            self.used_reactor_ids.extend(train.component_reactor_ids)
            return
        self._unresolved(
            operation,
            "reactors",
            "No available reactor or declared reactor train matches the required volume, ID, material, temperature, pressure, and system.",
            {"volume_mL": volume, "ID_mm": diameter, "material": material, "temperature_C": temperature},
        )

    def _matching_train(self, volume: float):
        reactor_by_id = {item.equipment_id: item for item in self.inventory.reactors}
        for train in sorted(self.inventory.reactor_trains, key=lambda item: item.equipment_id):
            if not _available(train) or self.remaining.get(train.equipment_id, 0) <= 0:
                continue
            components = [reactor_by_id.get(item_id) for item_id in train.component_reactor_ids]
            if not components or any(item is None for item in components):
                continue
            total = train.total_volume_mL or sum(item.volume_mL for item in components)
            if abs(total - volume) <= max(0.02, 0.005 * max(volume, 1.0)):
                if all(self.remaining.get(item.equipment_id, 0) > 0 for item in components):
                    return train
        return None

    def _claim_train(self, operation, train) -> bool:
        item_by_id = {item.equipment_id: item for item in self.inventory.all_equipment()}
        required_ids = [train.equipment_id, *train.component_reactor_ids, *train.connector_ids]
        if any(self.remaining.get(item_id, 0) <= 0 for item_id in required_ids):
            return False
        for item_id in required_ids:
            self.remaining[item_id] -= 1
        items = [item_by_id[item_id] for item_id in required_ids]
        self._apply_assignment(operation, "reactor_trains", items, {"configuration": "serial"})
        return True

    def _allocate_tubing(self) -> None:
        if self.proposal.stage_parameters:
            for index, stage in enumerate(self.proposal.stage_parameters, 1):
                self._allocate_tubing_path(
                    str(stage.get("reactor_equipment_id") or ""),
                    str(stage.get("material") or ""),
                    _num(stage.get("d_mm"), 0),
                    self.proposal.BPR_bar,
                    _num(stage.get("temperature_C"), self.proposal.temperature_C),
                    f"st{index}_reactor", f"assignment_tubing_stage_{index}",
                )
            return
        self._allocate_tubing_path(
            str((self.proposal.inventory_selection or {}).get("equipment_id") or ""),
            self.proposal.tubing_material, self.proposal.tubing_ID_mm,
            self.proposal.BPR_bar, self.proposal.temperature_C,
            "reactor_system", "assignment_tubing",
        )

    def _allocate_tubing_path(
        self, selected_id, material, diameter, pressure, temperature,
        operation_id, assignment_id,
    ) -> None:
        selected_reactor = next(
            (item for item in self.inventory.reactors if item.equipment_id == selected_id),
            None,
        )
        if selected_reactor is not None:
            identity = (
                f"{selected_reactor.type} {selected_reactor.configuration} "
                f"{selected_reactor.name}"
            ).lower()
            if any(
                token in identity
                for token in (
                    "microchannel", "microreactor", "packed-bed",
                    "packed bed", "integrated",
                )
            ):
                reactor_assignment = next(
                    (
                        item for item in self.assignments
                        if selected_reactor.equipment_id in item["equipment_item_ids"]
                    ),
                    None,
                )
                if reactor_assignment is not None:
                    reactor_assignment["settings"]["integrated_flow_path"] = True
                    reactor_assignment["capability_checks"]["integrated_flow_path"] = True
                return
        candidates = [
            item for item in self.inventory.tubing
            if _available(item)
            and item.material.lower() == material.lower()
            and abs(item.ID_mm - diameter) <= 1e-3
            and item.max_pressure_bar + 1e-12 >= pressure
            and item.max_temperature_C + 1e-12 >= temperature
        ]
        item = self._first_available(candidates)
        if item is None:
            if self.strict:
                self.unresolved.append(
                    {
                        "requirement_id": "INV-TUBING",
                        "operation_id": operation_id,
                        "category": "tubing",
                        "reason": "No available tubing matches material, ID, pressure, and temperature.",
                        "criteria": {
                            "material": material,
                            "ID_mm": diameter,
                            "pressure_bar": pressure,
                            "temperature_C": temperature,
                        },
                    }
                )
            return
        self.remaining[item.equipment_id] -= 1
        for key, amount in item.resource_requirements.items():
            self.resource_remaining[key] -= amount
        self.assignments.append(
            {
                "assignment_id": assignment_id,
                "operation_id": operation_id,
                "role": "reactor tubing",
                "category": "tubing",
                "equipment_item_ids": [item.equipment_id],
                "instrument_names": [item.name or f"{item.material} tubing"],
                "settings": {"material": item.material, "ID_mm": item.ID_mm},
                "capability_checks": {"available": True, "pressure": True, "temperature": True},
            }
        )

    def _attach_auxiliary_equipment(self, topology: ProcessTopology) -> None:
        reactors = [
            operation for operation in topology.unit_operations
            if operation.op_type.lower() in REACTOR_TYPES
        ]
        lights = [
            operation for operation in topology.unit_operations
            if operation.op_type.lower() == "led_module"
            and operation.assignment_status == "assigned"
        ]
        for light in lights:
            prefix = light.op_id.rsplit("_", 1)[0]
            reactor = next(
                (item for item in reactors if item.op_id.startswith(prefix)),
                reactors[0] if len(reactors) == 1 else None,
            )
            if reactor is None:
                continue
            reactor.parameters["light_instrument_name"] = light.instrument_name
            reactor.parameters["light_inventory_item_id"] = light.inventory_item_id
        controllers = [
            operation for operation in topology.unit_operations
            if operation.op_type.lower() in {"heat_exchanger", "heater", "chiller"}
            and operation.assignment_status == "assigned"
        ]
        for controller in controllers:
            prefix = controller.op_id.rsplit("_", 1)[0]
            reactor = next(
                (item for item in reactors if item.op_id.startswith(prefix)),
                reactors[0] if len(reactors) == 1 else None,
            )
            if reactor is None:
                continue
            reactor.parameters["temperature_controller_name"] = controller.instrument_name
            reactor.parameters["temperature_controller_inventory_item_id"] = controller.inventory_item_id

    def _validate_reactor_train(self, topology: ProcessTopology) -> None:
        """Validate only direct reactor-to-reactor physical connections.

        Two reaction stages separated by an assigned mixer are independent
        reactor operations. Standard stocked connectors also permit direct
        interstage links without enumerating every reactor pair. This does not
        synthesize a larger logical reactor from undeclared component coils.
        """

        if len(self.used_reactor_ids) <= 1:
            return
        reactor_operation_ids = {
            operation.op_id
            for operation in topology.unit_operations
            if operation.op_type.lower() in REACTOR_TYPES
        }
        direct_reactor_edges = [
            stream
            for stream in topology.streams
            if stream.from_op in reactor_operation_ids
            and stream.to_op in reactor_operation_ids
        ]
        if not direct_reactor_edges:
            return
        if any(self.inventory.capability_status.get(key) == "unavailable"
               for key in ("connectors", "connectors_or_reactor_trains")):
            if self.strict:
                self.unresolved.append({
                    "requirement_id": "INV-REACTOR-TRAIN",
                    "operation_id": "process_topology",
                    "category": "connectors",
                    "reason": "Required reactor connectors are explicitly unavailable.",
                    "criteria": {"component_reactor_ids": self.used_reactor_ids},
                })
            return
        connector_by_id = {item.equipment_id: item for item in self.inventory.connectors}
        declared = next(
            (
                train
                for train in self.inventory.reactor_trains
                if train.component_reactor_ids == self.used_reactor_ids
                and len(train.connector_ids) >= len(self.used_reactor_ids) - 1
                and _available(train)
                and self.remaining.get(train.equipment_id, 0) > 0
                and all(
                    item_id in connector_by_id
                    and _available(connector_by_id[item_id])
                    and self.remaining.get(item_id, 0) > 0
                    for item_id in train.connector_ids
                )
            ),
            None,
        )
        if declared is not None:
            self.remaining[declared.equipment_id] -= 1
            for connector_id in declared.connector_ids:
                self.remaining[connector_id] -= 1
            connector_items = [connector_by_id[item_id] for item_id in declared.connector_ids]
            self.assignments.append(
                {
                    "assignment_id": "assignment_reactor_train",
                    "operation_id": "process_topology",
                    "role": "serial reactor assembly",
                    "category": "reactor_trains",
                    "equipment_item_ids": [
                        declared.equipment_id, *declared.connector_ids
                    ],
                    "instrument_names": [
                        declared.name or declared.equipment_id,
                        *(item.name or item.equipment_id for item in connector_items),
                    ],
                    "settings": {
                        "configuration": "serial",
                        "component_reactor_ids": self.used_reactor_ids,
                    },
                    "capability_checks": {
                        "available": True,
                        "quantity": True,
                        "connection_order": True,
                    },
                }
            )
        elif self.inventory.allows_standard_reactor_connections:
            for edge in direct_reactor_edges:
                assumption_id = f"ASSUMED-CONNECTORS-{edge.stream_id}"
                name = "Standard interstage reactor connector"
                settings = {
                    "from_operation": edge.from_op,
                    "to_operation": edge.to_op,
                    "pressure_bar": self.proposal.BPR_bar,
                    "availability_basis": "standard_reactor_connectors_available",
                }
                self.assumed_accessories.append({
                    "assumption_id": assumption_id,
                    "operation_id": edge.stream_id,
                    "category": "connectors",
                    "name": name,
                    "settings": settings,
                    "requires_pre_run_verification": True,
                })
                self.assignments.append({
                    "assignment_id": f"assignment_connector_{edge.stream_id}",
                    "operation_id": edge.stream_id,
                    "role": "interstage reactor connection",
                    "category": "connectors",
                    "equipment_item_ids": [],
                    "instrument_names": [name],
                    "settings": settings,
                    "capability_checks": {
                        "passive_accessory_policy": True,
                        "requires_pre_run_verification": True,
                    },
                    "assumption_id": assumption_id,
                })
            self.warnings.append(
                "Standard interstage connectors are available by inventory policy; "
                "verify fittings, wetted materials, and pressure/temperature ratings before execution."
            )
        elif self.strict:
            reactor_operations = [
                operation
                for operation in self.compiled_operations
                if operation.op_type.lower() in REACTOR_TYPES
            ]
            if len(reactor_operations) > 1:
                reactor_operations[1].parameters[
                    "serial_connection_status"
                ] = "UNRESOLVED: serial reactor train/connectors"
            self.unresolved.append(
                {
                    "requirement_id": "INV-REACTOR-TRAIN",
                    "operation_id": "process_topology",
                    "category": "reactor_trains",
                    "reason": "Multiple reactors are used, but no compatible serial reactor train and connector set is declared.",
                    "criteria": {"component_reactor_ids": self.used_reactor_ids},
                }
            )

    def _claim(
        self,
        operation,
        category: str,
        candidates: Iterable[Any],
        settings: dict[str, Any],
        *,
        unresolved: bool = True,
    ) -> bool:
        item = self._first_available(candidates)
        if item is None:
            if unresolved:
                self._unresolved(
                    operation,
                    category,
                    f"No available {category.replace('_', ' ')} item satisfies the required settings.",
                    settings,
                )
            return False
        self.remaining[item.equipment_id] -= 1
        self._apply_assignment(operation, category, [item], settings)
        for key, amount in item.resource_requirements.items():
            self.resource_remaining[key] -= amount
        return True

    def _apply_assignment(self, operation, category: str, items: list[Any], settings: dict[str, Any]) -> None:
        ids = [item.equipment_id for item in items]
        names = [item.name or item.equipment_id for item in items]
        operation.inventory_item_id = ids[0]
        operation.inventory_item_ids = ids
        operation.instrument_name = " + ".join(names)
        operation.inventory_category = category
        operation.assignment_status = "assigned"
        operation.capability_checks = {"available": True, "quantity": True, "settings": True}
        operation.parameters.update(
            {
                "inventory_item_id": ids[0],
                "inventory_item_ids": ids,
                "instrument_name": operation.instrument_name,
            }
        )
        operation.label = f"{operation.label} | {operation.instrument_name}"
        self.assignments.append(
            {
                "assignment_id": f"assignment_{operation.op_id}",
                "operation_id": operation.op_id,
                "role": operation.label.split(" | ", 1)[0],
                "category": category,
                "equipment_item_ids": ids,
                "instrument_names": names,
                "settings": settings,
                "capability_checks": operation.capability_checks,
            }
        )

    def _assume_standard_accessory(
        self,
        operation,
        category: str,
        settings: dict[str, Any],
    ) -> None:
        """Resolve an undeclared passive fitting without inventing a lab asset."""

        assumption_id = f"ASSUMED-{category.upper()}-{operation.op_id}"
        name = "Generic compatible T-mixer - verify before run"
        operation.inventory_item_id = None
        operation.inventory_item_ids = []
        operation.instrument_name = name
        operation.inventory_category = category
        operation.assignment_status = "assumed_standard_accessory"
        operation.capability_checks = {
            "inventory_declared": False,
            "passive_accessory_policy": True,
            "requires_pre_run_verification": True,
        }
        operation.parameters.update(
            {
                "instrument_name": name,
                "inventory_assignment_status": "assumed_standard_accessory",
                "pre_run_verification_required": True,
            }
        )
        operation.label = f"{operation.label} | Generic T-mixer [VERIFY]"
        assumption = {
            "assumption_id": assumption_id,
            "operation_id": operation.op_id,
            "category": category,
            "name": name,
            "settings": settings,
            "requires_pre_run_verification": True,
        }
        self.assumed_accessories.append(assumption)
        self.assignments.append(
            {
                "assignment_id": f"assignment_{operation.op_id}",
                "operation_id": operation.op_id,
                "role": operation.label.split(" | ", 1)[0],
                "category": category,
                "equipment_item_ids": [],
                "instrument_names": [name],
                "settings": settings,
                "capability_checks": operation.capability_checks,
                "assumption_id": assumption_id,
            }
        )
        self.warnings.append(
            f"{operation.op_id}: using an undeclared generic T-mixer under the "
            "standard passive-accessory policy; verify compatibility before the run."
        )

    def _unresolved(
        self,
        operation,
        category: str,
        reason: str,
        criteria: dict[str, Any] | None = None,
    ) -> None:
        if not self.strict:
            operation.assignment_status = "legacy_untracked"
            operation.inventory_category = category
            self.warnings.append(f"{operation.op_id}: {reason}")
            return
        operation.assignment_status = "unresolved"
        operation.inventory_category = category
        operation.parameters["inventory_assignment_status"] = "unresolved"
        operation.parameters["unresolved_inventory_category"] = category
        self.unresolved.append(
            {
                "requirement_id": f"INV-{category.upper().replace('_', '-')}",
                "operation_id": operation.op_id,
                "category": category,
                "reason": reason,
                "criteria": criteria or {},
            }
        )

    def _first_available(self, candidates: Iterable[Any]):
        return next(
            (item for item in candidates if self.remaining.get(item.equipment_id, 0) > 0
             and all(self.resource_remaining.get(key, 0) >= amount for key, amount in item.resource_requirements.items())),
            None,
        )

    def _insert_required_gas_accessories(self, topology):
        from flora_translate.schemas import UnitOperation
        devices = {x.equipment_id: x for x in self.inventory.gas_hardware}
        for op in list(topology.unit_operations):
            if op.op_type.lower() != "mfc":
                continue
            device = devices.get(op.parameters.get("inventory_equipment_id"))
            if not device or device.required_outlet_accessory_type != "check_valve":
                continue
            downstream = [edge for edge in topology.streams if edge.from_op == op.op_id and edge.connection_type == "process"]
            if not downstream:
                continue
            if all(any(x.op_id == edge.to_op and x.op_type == "check_valve" for x in topology.unit_operations) for edge in downstream):
                continue
            valve_id = op.op_id + "_check_valve"
            topology.unit_operations.insert(topology.unit_operations.index(op) + 1, UnitOperation(
                op_id=valve_id, op_type="check_valve", label="Check valve", required=True,
                rationale="Required by gas-delivery inventory to prevent pressurized-liquid backflow.",
                parameters={"direction": "toward reactor"}))
            first = downstream[0].model_copy(deep=True)
            first.stream_id += "_to_check_valve"
            first.to_op = valve_id
            for edge in downstream:
                edge.from_op = valve_id
            topology.streams.append(first)

    def _remove_redundant_mixers(self, topology: ProcessTopology) -> None:
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for stream in topology.streams:
            incoming[stream.to_op].append(stream)
            outgoing[stream.from_op].append(stream)
        remove_ids = {
            operation.op_id
            for operation in topology.unit_operations
            if operation.op_type.lower() in MIXER_TYPES
            and len(incoming.get(operation.op_id, [])) <= 1
        }
        for operation_id in remove_ids:
            predecessors = incoming.get(operation_id, [])
            successors = outgoing.get(operation_id, [])
            if len(predecessors) == 1:
                predecessor = predecessors[0].from_op
                for successor in successors:
                    successor.from_op = predecessor
        topology.unit_operations = [
            operation for operation in topology.unit_operations if operation.op_id not in remove_ids
        ]
        topology.streams = [
            stream for stream in topology.streams
            if stream.from_op not in remove_ids and stream.to_op not in remove_ids
        ]

    def _instrument_manifest(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for assignment in self.assignments:
            for item_id, name in zip(
                assignment["equipment_item_ids"], assignment["instrument_names"]
            ):
                entry = grouped.setdefault(
                    item_id,
                    {
                        "equipment_id": item_id,
                        "name": name,
                        "category": assignment["category"],
                        "quantity_used": 0,
                        "roles": [],
                        "settings": [],
                    },
                )
                entry["quantity_used"] += 1
                entry["roles"].append(assignment["role"])
                entry["settings"].append(assignment["settings"])
        manifest = list(grouped.values())
        manifest.extend(
            {
                "equipment_id": item["assumption_id"],
                "name": item["name"],
                "category": item["category"],
                "quantity_used": 1,
                "roles": [item["operation_id"]],
                "settings": [item["settings"]],
                "assignment_status": "assumed_standard_accessory",
                "requires_pre_run_verification": True,
            }
            for item in self.assumed_accessories
        )
        return manifest

    def _no_unknown_ids(self) -> bool:
        known = {item.equipment_id for item in self.inventory.all_equipment()}
        assigned = {
            item_id
            for assignment in self.assignments
            for item_id in assignment["equipment_item_ids"]
        }
        return assigned <= known


def _incoming_counts(topology: ProcessTopology) -> Counter:
    return Counter(stream.to_op for stream in topology.streams)


def _inventory_sha256(inventory: LabInventory) -> str:
    payload = json.dumps(
        inventory.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _available(item) -> bool:
    return str(item.service_status or "available").strip().lower() in AVAILABLE_STATUSES


def _pressure_supported(item, pressure: float) -> bool:
    if item.setpoints_bar:
        return any(abs(value - pressure) <= 0.05 for value in item.setpoints_bar)
    return (
        (item.min_pressure_bar is None or pressure >= item.min_pressure_bar - 1e-12)
        and (item.max_pressure_bar is None or pressure <= item.max_pressure_bar + 1e-12)
    )


def _temperature_supported(item, temperature: float) -> bool:
    if item.allowed_temperatures_C:
        return any(abs(value - temperature) <= 0.1 for value in item.allowed_temperatures_C)
    return (
        (item.min_temperature_C is None or temperature >= item.min_temperature_C - 1e-12)
        and (item.max_temperature_C is None or temperature <= item.max_temperature_C + 1e-12)
    )


def _light_temperature_supported(item, temperature: float) -> bool:
    if item.allowed_temperatures_C:
        return any(abs(value - temperature) <= 0.1 for value in item.allowed_temperatures_C)
    return (
        (item.min_temperature_C is None or temperature >= item.min_temperature_C - 1e-12)
        and (item.max_temperature_C is None or temperature <= item.max_temperature_C + 1e-12)
    )


def _reactor_temperature_supported(item, temperature: float) -> bool:
    if item.allowed_temperatures_C:
        return any(abs(value - temperature) <= 0.1 for value in item.allowed_temperatures_C)
    return (
        (item.min_temperature_C is None or temperature >= item.min_temperature_C - 1e-12)
        and (item.max_temperature_C is None or temperature <= item.max_temperature_C + 1e-12)
    )


def _operation_gas(operation) -> str:
    contents = operation.parameters.get("contents") or []
    return " ".join(str(value) for value in contents)


def _gas_matches(required: str, available: str) -> bool:
    aliases = {
        "h2": "hydrogen",
        "hydrogen": "hydrogen",
        "o2": "oxygen",
        "oxygen": "oxygen",
        "n2": "nitrogen",
        "nitrogen": "nitrogen",
        "air": "air",
    }

    def identities(value: str) -> set[str]:
        normalized = value.lower().replace("₂", "2")
        tokens = set(re.findall(r"[a-z]+\d*", normalized))
        return {aliases.get(token, token) for token in tokens}

    return bool(identities(required) & identities(available))


def _pump_material_compatible(pump, stream_identity: str) -> bool:
    """Require a declared wetted-material match when a pump lists one."""

    if not pump.compatible_materials:
        return True
    identity = " ".join(re.findall(r"[a-z0-9]+", stream_identity.lower()))
    for material in pump.compatible_materials:
        candidate = " ".join(re.findall(r"[a-z0-9]+", str(material).lower()))
        candidate_tokens = {
            token for token in candidate.split()
            if token not in {"aqueous", "solution", "feed"}
        }
        identity_tokens = set(identity.split())
        if candidate and (
            candidate in identity
            or (candidate_tokens and candidate_tokens <= identity_tokens)
        ):
            return True
    return False


def _num(value: Any, default: Any = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(default)
        except (TypeError, ValueError):
            return 0.0
