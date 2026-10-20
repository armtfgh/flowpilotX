"""Inventory-backed process-topology compiler."""

from __future__ import annotations

from flora_translate.inventory_allocator import InventoryAllocator
from flora_translate.schemas import FlowProposal, LabInventory, ProcessTopology


def compile_inventory_topology(
    topology: ProcessTopology,
    *,
    proposal: FlowProposal,
    inventory: LabInventory | None,
) -> tuple[ProcessTopology, dict]:
    """Return a topology whose physical operations reference inventory items."""

    if inventory is None:
        report = {
            "schema_version": "flowpilot_inventory_allocation_v1.0",
            "status": "not_applied",
            "strict_assignment": False,
            "checks": {"all_required_operations_assigned": True},
            "assignments": [],
            "instrument_manifest": [],
            "unresolved_requirements": [],
            "warnings": ["No laboratory inventory was supplied."],
        }
        return topology.model_copy(deep=True), report
    return InventoryAllocator(inventory, proposal).compile(topology)
