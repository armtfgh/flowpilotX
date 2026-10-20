from pathlib import Path
import pytest
from flora_translate.inventory_profiles import InventoryProfile
from flora_translate.schemas import LabInventory,ReactorSpec
from scripts.build_khu_inventory_v6 import corrected_profile


def source():
    return InventoryProfile(profile_id='test',name='test',lab_inventory=LabInventory(reactors=[
        ReactorSpec(equipment_id='coil10',type='coil',material='PFA',volume_mL=10,ID_mm=1,quantity=1)]))


def definition(**changes):
    return dict(material='PFA',volume_mL=10,ID_mm=1,max_pressure_bar=8,quantity=1,
                source='Synthetic fixture only, not laboratory confirmation',separate_from_manual_stock=True,**changes)


def test_separate_assembly_keeps_manual_stock_and_binds_membership():
    before=source();snapshot=before.model_dump_json()
    result=corrected_profile(before,definition())
    assert before.model_dump_json()==snapshot
    assert result.lab_inventory.reactors[0].photoreactor_module_ids==['manual1','manual2']
    assert result.lab_inventory.reactors[1].photoreactor_module_ids==['uv150']


def test_shared_coil_is_not_counted_twice():
    data=definition(shared_with_manual_equipment_id='coil10');data['separate_from_manual_stock']=False
    result=corrected_profile(source(),data)
    assert result.lab_inventory.reactors[0].service_status=='integrated_component'


def test_no_installed_reactor_is_invented_from_capacity():
    with pytest.raises(ValueError,match='Missing confirmed'):
        corrected_profile(source(),{'volume_mL':10})


def test_wrong_shared_coil_is_rejected():
    data=definition(shared_with_manual_equipment_id='unknown');data['separate_from_manual_stock']=False
    with pytest.raises(ValueError,match='exactly match'):
        corrected_profile(source(),data)
