from types import SimpleNamespace as NS

from flora_translate.equipment_resources import resources_fit, light_fits_stage
from flora_translate.schemas import LabInventory, LightSourceSpec, ReactorSpec, FlowProposal, PumpSpec, StreamAssignment
from flora_translate.intake_agent import _gas_introduction_stage


def test_shared_module_and_platform_resources():
    a=NS(resource_requirements={'uv150':1})
    assert resources_fit([a],{'uv150':1})
    assert not resources_fit([a,a],{'uv150':1})
    assert not resources_fit([a],{})


def test_assembly_limits_not_just_wavelength():
    light=LightSourceSpec(wavelength_nm=450,compatible_reactor='coil',module_id='uv150',module_name='Vapourtec UV-150',
                         max_reactor_volume_mL=10,compatible_pump_platforms=['e_series'])
    r=ReactorSpec(type='coil',material='PFA',volume_mL=10,ID_mm=1)
    pump=PumpSpec(equipment_id='p',type='peristaltic',platform_id='e_series',min_flow_rate_mL_min=.02,
                  max_flow_rate_mL_min=10,max_pressure_bar=10)
    inventory=LabInventory(pumps=[pump])
    proposal=FlowProposal(streams=[StreamAssignment(stream_label='A',pump_equipment_id='p')])
    assert not light_fits_stage(light,r,proposal,inventory)
    r=r.model_copy(update={'system':'uv150','photoreactor_module_ids':['uv150']})
    assert light_fits_stage(light,r,proposal,inventory)
    assert not light_fits_stage(light,r.model_copy(update={'system':'manual1','photoreactor_module_ids':['manual1']}),proposal,inventory)
    assert not light_fits_stage(light,r.model_copy(update={'volume_mL':20}),proposal,inventory)
    assert not light_fits_stage(light,r,FlowProposal(),inventory)


def test_manual_coil_cannot_borrow_integrated_light_even_with_matching_pump():
    light=LightSourceSpec(wavelength_nm=450,compatible_reactor='coil',module_id='uv150',
                         module_name='Vapourtec UV-150',max_reactor_volume_mL=10)
    manual=ReactorSpec(type='coil',material='PFA',volume_mL=2,ID_mm=1,system='manual',
                       photoreactor_module_ids=['manual1','manual2'])
    assert not light_fits_stage(light,manual,FlowProposal(),LabInventory())
    for module in ['manual1','manual2']:
        assert light_fits_stage(light.model_copy(update={'module_id':module}),manual,FlowProposal(),LabInventory())
    assert not light_fits_stage(light.model_copy(update={'module_id':''}),manual,FlowProposal(),LabInventory())


def test_integrated_reactor_cannot_borrow_manual_light():
    reactor=ReactorSpec(type='coil',material='PFA',volume_mL=10,ID_mm=1,system='uv150',
                        photoreactor_module_ids=['uv150'])
    manual=LightSourceSpec(wavelength_nm=448,compatible_reactor='coil',module_id='manual2')
    assert not light_fits_stage(manual,reactor,FlowProposal(),LabInventory())


def test_module_membership_round_trip():
    reactor=ReactorSpec(type='coil',material='PFA',volume_mL=2,ID_mm=1,
                        photoreactor_module_ids=['manual1','manual2'])
    assert ReactorSpec.model_validate_json(reactor.model_dump_json()).photoreactor_module_ids==['manual1','manual2']


def test_positive_gas_destination_with_exclusion():
    assert _gas_introduction_stage('Introduce oxygen only at the Stage 2 inlet. Stage 1 must remain oxygen-free.') == 2
    assert _gas_introduction_stage('Introduce oxygen at Stage 1. Stage 1 must remain oxygen-free.') is None


def test_neutral_pump_asset():
    from flora_design.visualizer.flowsheet_builder import _prepare_pump_img
    from pathlib import Path
    p=_prepare_pump_img()
    assert p.name == '_pump_generic.png' and p.stat().st_size>1000
    assert 'Syringe' not in Path(p.parent/'pump_generic.svg').read_text()


def test_declared_buffer_medium_does_not_get_substrate_molarity():
    from flora_translate.component_identity import is_solvent_component
    from flora_translate.design_realizer import _resolve_component_quantity_assumptions
    assert is_solvent_component('pH 9 aqueous buffer','EtOH:pH 9 buffer 5:1 v/v','unknown')
    assert is_solvent_component('pH 9 aqueous buffer','EtOH:pH 9 buffer 5:1 v/v','co-solvent / base')
    assert is_solvent_component('pH 9 aqueous buffer','EtOH:pH 9 buffer (5:1 v/v), degassed under Ar','co-solvent / base')
    p=FlowProposal(streams=[StreamAssignment(stream_label='A',contents=['pH 9 aqueous buffer'],solvent='EtOH:pH 9 buffer 5:1 v/v',concentration_M=.1)])
    decisions=[]
    _resolve_component_quantity_assumptions(p,decisions)
    assert not decisions
    assert p.streams[0].contents==['pH 9 aqueous buffer']


def test_inventory_required_check_valve_is_inserted():
    from flora_translate.inventory_allocator import InventoryAllocator
    from flora_translate.schemas import GasHardwareSpec, SafetyAccessorySpec, ProcessTopology, UnitOperation, StreamConnection
    inventory=LabInventory(gas_hardware=[GasHardwareSpec(equipment_id='mfc',type='MFC',gas='O2',
        required_outlet_accessory_type='check_valve')], safety_accessories=[SafetyAccessorySpec(equipment_id='cv',type='check_valve',cracking_pressure_bar=1)])
    topology=ProcessTopology(unit_operations=[UnitOperation(op_id='g',op_type='mfc',parameters={'inventory_equipment_id':'mfc'}),
        UnitOperation(op_id='mix',op_type='mixer')],streams=[StreamConnection(from_op='g',to_op='mix',stream_type='gas')])
    allocator=InventoryAllocator(inventory,FlowProposal())
    allocator._insert_required_gas_accessories(topology)
    allocator._insert_required_gas_accessories(topology)
    assert len([o for o in topology.unit_operations if o.op_type=='check_valve'])==1
    assert {(s.from_op,s.to_op) for s in topology.streams}=={('g','g_check_valve'),('g_check_valve','mix')}


def test_topology_uses_final_feed_concentration():
    from flora_translate.main import _build_translate_topology
    from flora_translate.schemas import BatchRecord
    p=FlowProposal(streams=[StreamAssignment(stream_label='A',contents=['Example reagent (1.0 equiv, neat or in 2-MeTHF)'],
        solvent='2-MeTHF',concentration_M=2.1,flow_rate_mL_min=.03)],flow_rate_mL_min=.03,reactor_volume_mL=3,residence_time_min=100)
    topology=_build_translate_topology(p,None,BatchRecord())
    feed=next(o for o in topology.unit_operations if o.op_type=='pump')
    assert feed.parameters['concentration_M']==2.1
    assert 'neat or' not in str(feed.parameters['contents'])
    assert '2.1 M' in str(feed.parameters['contents'])


def test_air_oxidant_role_does_not_imply_pure_oxygen():
    from flora_translate.final_design_contract import _gas_composition_and_equivalent_issues
    gas={'phase':'gas','contents':['air'],'pump_role':'Air gas feed providing O2 as terminal oxidant',
         'gas_reagent_mole_fraction':.21}
    assert not _gas_composition_and_equivalent_issues({'streams':[gas]})
    gas['contents']=['O2']
    assert _gas_composition_and_equivalent_issues({'streams':[gas]})[0]['code']=='FINAL-GAS-COMPOSITION-INCONSISTENT'
