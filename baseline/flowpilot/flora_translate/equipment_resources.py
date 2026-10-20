"""Shared physical resources and stage-local photoreactor compatibility."""

from collections import Counter


def resources_fit(items, capacities):
    used = Counter()
    for item in items:
        used.update(item.resource_requirements)
    return all(key in capacities and count <= capacities[key] for key, count in used.items())


def light_fits_stage(light, reactor, proposal, inventory):
    if light.max_reactor_volume_mL is not None and reactor.volume_mL > light.max_reactor_volume_mL + 1e-9:
        return False
    # Module membership must be explicit. A pump platform or free volume does
    # not establish that a loose/manual coil belongs to an integrated reactor.
    modules = {str(key).strip().lower() for key in reactor.photoreactor_module_ids}
    module = str(light.module_id or "").strip().lower()
    system = str(reactor.system or "").strip().lower()
    if modules:
        if not module or module not in modules:
            return False
    elif module:
        if not system or system not in {module, str(light.module_name or "").strip().lower()}:
            return False
    if light.compatible_pump_platforms:
        pumps = {p.equipment_id: p for p in inventory.pumps}
        platforms = {pumps[s.pump_equipment_id].platform_id for s in proposal.streams
                     if s.phase != "gas" and s.pump_equipment_id in pumps}
        if not platforms.intersection(light.compatible_pump_platforms):
            return False
    return True


def pump_accepts_feed(pump, stream):
    text = " ".join([*stream.contents, stream.solvent, stream.pump_role]).lower()
    return not any(chemical.lower() in text for chemical in pump.excluded_chemicals)
