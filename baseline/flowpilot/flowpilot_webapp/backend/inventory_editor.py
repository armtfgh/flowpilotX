"""Read-only form metadata derived from the pipeline's inventory models."""

from flora_translate.inventory_profiles import empty_inventory_profile
from flora_translate.schemas import LabInventory


LABELS = {
    "pumps": "Pumps", "reactors": "Reactors", "tubing": "Tubing",
    "light_sources": "Light sources", "gas_hardware": "Gas hardware",
    "mixers": "Mixers", "pressure_controllers": "Pressure controllers",
    "temperature_controllers": "Temperature controllers", "degassers": "Degassers",
    "filters": "Filters", "separators": "Separators", "connectors": "Connectors",
    "collectors": "Collectors", "reactor_trains": "Reactor trains",
    "safety_accessories": "Safety accessories",
}


def editor_schema() -> dict:
    schema = LabInventory.model_json_schema()
    categories = []
    for key, label in LABELS.items():
        reference = schema["properties"][key]["items"]["$ref"].split("/")[-1]
        categories.append({"key": key, "label": label, "schema": schema["$defs"][reference]})
    return {
        "categories": categories,
        "inventory_schema": schema,
        "empty_profile": empty_inventory_profile().model_dump(mode="json"),
    }
