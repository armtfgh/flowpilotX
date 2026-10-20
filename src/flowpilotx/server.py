"""Loopback campaign application; physical operation and publication are out of scope."""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit, unquote
from uuid import uuid4

from .campaign import CampaignStore
from .review_import import ARCHIVED_ROUTE, registered_asset
from .gateway import ModelGateway, PLANNER_SYSTEM, planning_context, validate_plan, write
from .demo import build_demo_profile, empty_demo_history, portable_requirements


# A partial study bundle is an error, never silently relabelled as a demo.
KHU_REQUIRED_FILES = (
    'KHU_inventory_complete_20260915.xlsx', 'KHU_inventory_complete_20260918_v7.json',
    'configs/khu_flowpilotx_2026-10-15_v1.json',
    'data/private/khu_source_context_v1.json',
    'data/raw/inventory/source_review_20260915/inventory_source_cells.md',
    'data/raw/inventory/source_review_20260915/khu_clarification.md',
    'data/raw/khu/1. Selective oxidation of sulfide_optmization table and substrate scope.pptx',
    'data/raw/khu/2-1.sulfide oxidation_raw data_homogeneous.xlsx',
    'data/raw/khu/2-2.sulfide oxidation_raw data_homogeneous.csv',
    'data/raw/khu/3-1.sulfide oxidation_raw data_heterogeneous.xlsx',
    'data/raw/khu/3-2.sulfide oxidation_raw data_heterogeneous.csv',
    'docs/INVENTORY_SOURCES.md', 'provenance/khu_lab_reply_reconciliation_2026-10-15_v1.json',
    'review/literature_2026-10-13_v1/assessment.json', 'review/slides/slide-2.png', 'review/slides/slide-3.png',
)


def resolve_profile(root, requested):
    if requested not in {'auto', 'demo', 'khu'}:
        raise ValueError('Profile must be auto, demo or khu')
    if requested == 'demo':
        return 'demo'
    markers = ('KHU_inventory_complete_20260918_v7.json', 'KHU_inventory_complete_20260915.xlsx', 'data/raw/khu')
    if requested == 'auto' and not any((root / path).exists() for path in markers):
        return 'demo'
    missing = [path for path in KHU_REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        raise ValueError('KHU profile requires the complete private source bundle. Missing: ' + ', '.join(missing)
                         + '. Use --profile demo for an independent synthetic demonstration.')
    return 'khu'


class Application:
    def __init__(self, project_root, storage_root=None, *, profile='auto'):
        self.root=Path(project_root).resolve()
        self.profile=resolve_profile(self.root, profile)
        default_storage = self.root / ('outputs/flowpilotx_platform_2026-10-15_v1' if self.profile == 'khu' else '.flowpilotx')
        self.storage=Path(storage_root or default_storage).resolve()
        self.store=CampaignStore(self.storage/'campaigns')
        max_calls=12 if self.profile == 'khu' else int(os.environ.get('FLOWPILOTX_QWEN_MAX_CALLS','0'))
        if max_calls < 0:
            raise ValueError('FLOWPILOTX_QWEN_MAX_CALLS must be a nonnegative integer')
        self.gateway=ModelGateway(self.storage/'model_calls',max_calls=max_calls)
        self.static=Path(__file__).resolve().parent/'gui'
        self.lock=threading.RLock()
        self._history=None

    def builtin(self):
        if self._history is None:
            if self.profile == 'khu':
                from khu_sulfoxidation.gui_data import builtin_history
                self._history=builtin_history(self.root)
            else:
                self._history=empty_demo_history()
        return self._history

    def ensure_default(self):
        pointer=self.storage/('default_campaign.json' if self.profile == 'khu' else 'default_demo_campaign.json')
        if pointer.exists(): return json.loads(pointer.read_text())['id']
        if self.profile == 'khu':
            from khu_sulfoxidation.flowpilotx_profile import build_profile
            spec=build_profile(self.root)
        else:
            spec=build_demo_profile()
        campaign=self.store.create(spec)
        ident=campaign['id']
        write(pointer,{'id':ident})
        selected=spec['default_selected_candidate_ids']
        self.store.generate(ident,{'requested_count':len(selected),'repeat_count':1,'repeat_of':spec['optional_repeat']['repeat_of'],
            'allow_conditional':True,'model_route':'deterministic',
            'selected_candidate_ids':selected,
            'selection_method':'synthetic demonstration coverage' if self.profile == 'demo' else 'engineering-checked exploratory screen'})
        return ident

    def requirements(self):
        if self.profile == 'demo': return portable_requirements()
        p=self.storage/'requirements_status.json'
        p=p if p.exists() else self.root/'configs/flowpilotx_requirements_v1.json'
        return json.loads(p.read_text()) if p.exists() else portable_requirements()

    def benchmark(self):
        p=self.storage/'benchmark_report.json'
        report=json.loads(p.read_text()) if p.exists() else {'status':'not_run','checks':[],
              'scope':'Software and model compliance; chemistry performance unmeasured'}
        return {**report,'model_gateway':self.gateway.status()}

    def bootstrap(self):
        ident=self.ensure_default()
        builtin=self.builtin()
        model_route=deepcopy(self.gateway.route)
        model_available=bool(model_route.get('available'))
        return {'application':'FlowPilotX','version':'0.2.1','profile':self.profile,'synthetic_demo':self.profile=='demo','campaigns':self.store.list(),
                'default_campaign_id':ident,'default_model_route':'deterministic',
                'models':[model_route,ARCHIVED_ROUTE,{'id':'deterministic','label':'Offline coverage selection','available':True,'paid':False}],
                'requirements':self.requirements(),'benchmark':self.benchmark(),
                'history':{k:deepcopy(builtin[k]) for k in ['id','rows','sources','warnings','summary']},
                'capabilities':{'model_generation':model_available,'results_feedback':True,'generic_inventory':True,
                   'generic_candidate_pool':True,'bo':False,'calibrated_information_gain':False,
                   'automatic_equipment_control':False,'paid_models':False,
                   'arbitrary_chemistry_candidate_synthesis':False,'builtin_history':self.profile=='khu'}}

    def generate(self, ident, request):
        request=deepcopy(request)
        if 'instructions' in request:
            request.setdefault('researcher_instructions',request.pop('instructions'))
        request['requested_count']=request.get('requested_count',request.get('count',6))
        route=request.get('model_route','deterministic')
        campaign=self.store.get(ident)
        spec=campaign.get('spec',campaign)
        mode=request.pop('objective_mode',None)
        if mode and mode != spec.get('objective',{}).get('mode'):
            campaign=self.store.revise(ident,{'objective':{**spec.get('objective',{}),'mode':mode}})
        request['expected_campaign_version']=campaign['version']
        request['expected_revision_sha256']=campaign['content_sha256']
        if route=='qwen3.6-27b':
            result=self.gateway.call('InteractivePlanner',PLANNER_SYSTEM,planning_context(campaign,request))
            evaluation=validate_plan(result['parsed'],campaign,request['requested_count'])
            write(Path(result['archive'])/'validation.json',evaluation)
            if not evaluation['passed']:
                raise ValueError('Model proposal failed validation; archived without fallback: '+ '; '.join(evaluation['errors']))
            request.update(selected_candidate_ids=result['parsed']['selected_candidate_ids'],model_review=result,
                           selection_method='Qwen scientific planning with deterministic physical gates')
        elif route!='deterministic':
            raise ValueError('Route unavailable in this campaign; no paid or hidden model fallback')
        result=self.store.generate(ident,request)
        return result

    def import_history(self, request):
        raw=base64.b64decode(request['content_base64'],validate=True)
        if not raw or len(raw)>8*1024*1024: raise ValueError('Upload must contain 1 byte to8MiB')
        source_sha=hashlib.sha256(raw).hexdigest()
        ident='import-'+uuid4().hex[:16]
        folder=self.storage/'imports'/ident
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'source.bin').write_bytes(raw)
        write(folder/'request.json',{'name':request['name'],'source_sha256':source_sha})
        try:
            known=self.builtin()['known_sources']
            if source_sha in known:
                from khu_sulfoxidation.gui_data import parse_history_upload
                parsed=parse_history_upload(request['name'],raw,known_sources=known)
                records=parsed['rows']
                for row in records: row.setdefault('id',row['record_id'])
            else:
                from .evidence import import_evidence
                parsed=import_evidence(request['name'],raw)
                records=parsed.get('records',parsed.get('rows',[]))
            result={**parsed,'id':ident,'records':records,'source_sha256':source_sha}
            if request.get('campaign_id'):
                campaign=self.store.get(request['campaign_id'])
                spec=campaign.get('spec',campaign)
                current_evidence=spec.get('evidence',[])
                if isinstance(current_evidence,dict):
                    evidence_rows=current_evidence.get('records',current_evidence.get('rows',[]))
                else:evidence_rows=current_evidence
                all_records={r.get('id',r.get('record_id')):deepcopy(r) for r in evidence_rows}
                conflicts=[]
                for row in records:
                    key=row.get('id',row.get('record_id'))
                    if key in all_records:
                        existing=all_records[key]
                        # Preserve alternative export representations, never sum observations.
                        existing.setdefault('import_representations',[]).append(deepcopy(row))
                        differences=[]
                        for canonical,alternative in [('reported_actual_conditions','reported_actual_conditions'),
                                                       ('outcomes','reported_outcomes'),('outcomes','outcomes')]:
                            left,right=existing.get(canonical),row.get(alternative)
                            if isinstance(left,dict) and isinstance(right,dict):
                                differences.extend(canonical+'.'+k for k in left.keys()&right.keys()
                                    if not isinstance(left[k],(dict,list)) and left[k]!=right[k])
                        if differences and source_sha not in known:
                            existing['interpretation_eligible']=False
                            existing['unresolved_import_conflicts']=sorted(set(existing.get('unresolved_import_conflicts',[])+differences))
                            conflicts.append({'record_id':key,'fields':differences,'policy':'Both reports preserved; no silent resolution or averaging'})
                        elif differences:
                            existing.setdefault('recognized_export_reconciliation',[]).append({
                                'source_sha256':source_sha,'fields':differences,
                                'policy':'Recognized original export: preserve established canonical reconciliation; rounding differences are not new conflicts'})
                    else: all_records[key]=row
                replacement=list(all_records.values())
                if isinstance(current_evidence,dict):
                    replacement={**current_evidence,'records':replacement}
                self.store.revise(request['campaign_id'],{'evidence':replacement})
                result['linked_campaign_id']=request['campaign_id']
                result['conflicts']=conflicts
            write(folder/'result.json',result)
            return result
        except Exception as exc:
            write(folder/'error.json',{'error':str(exc),'original_preserved':True})
            raise

    def round(self, ident, round_id):
        for r in self.store.get(ident).get('rounds',[]):
            if r.get('id',r.get('round_id'))==round_id:return r
        raise ValueError('Round not found')


def handler(application):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,obj,status=200,mime='application/json'):
            data=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode() if mime=='application/json' else obj
            self.send_response(status)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(data)
        def valid_origin(self):
            host=self.headers.get('Host','')
            if not re.fullmatch(r'(127\.0\.0\.1|localhost)(:\d+)?',host):return False
            origin=self.headers.get('Origin')
            return origin is None or urlsplit(origin).netloc==host
        def body(self):
            if not self.valid_origin():raise ValueError('Only same-origin loopback requests are accepted')
            length=int(self.headers.get('Content-Length',0))
            if length<=0 or length>12*1024*1024:raise ValueError('Invalid request length')
            obj=json.loads(self.rfile.read(length))
            if not isinstance(obj,dict):raise ValueError('JSON object required')
            return obj
        def do_GET(self):
            try:
                if not self.valid_origin():raise ValueError('Invalid host')
                path=unquote(urlsplit(self.path).path)
                if path=='/api/bootstrap':return self.send(application.bootstrap())
                if path=='/api/requirements':return self.send(application.requirements())
                if path=='/api/benchmark':return self.send(application.benchmark())
                if path=='/api/campaigns':return self.send(application.store.list())
                match=re.fullmatch(r'/api/campaigns/([A-Za-z0-9_-]+)',path)
                if match:return self.send(application.store.get(match[1]))
                match=re.fullmatch(r'/api/rounds/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)',path)
                if match:return self.send(application.round(*match.groups()))
                match=re.fullmatch(r'/assets/packages/([A-Za-z0-9_-]+)/(.+)',path)
                if match:
                    file=registered_asset(application.root,application.storage,*match.groups())
                    return self.send(file.read_bytes(),mime=mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                if path.startswith('/assets/icons/'):
                    base=application.root/'baseline/flowpilot/flora_design/visualizer/icons';rel=path[len('/assets/icons/'):]
                elif path.startswith('/assets/khu/'):
                    base=application.root/'outputs/flowpilotx_khu_2026-10-15_v1';rel=path[len('/assets/khu/'):]
                else:base=application.static;rel='index.html' if path=='/' else path.lstrip('/')
                file=(base/rel).resolve()
                if not file.is_relative_to(base.resolve()) or not file.is_file():return self.send({'error':'Not found'},404)
                return self.send(file.read_bytes(),mime=mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            except Exception as exc:self.send({'error':str(exc)},400)
        def mutate(self,method):
            try:
                path=urlsplit(self.path).path;obj=self.body()
                if path=='/api/history/import':return self.send(application.import_history(obj))
                if path=='/api/campaigns' and method=='POST':return self.send(application.store.create(obj),201)
                match=re.fullmatch(r'/api/campaigns/([A-Za-z0-9_-]+)(?:/(generate|results|questions))?',path)
                if not match:return self.send({'error':'Not found'},404)
                ident,action=match.groups()
                if method=='PATCH' and action is None:return self.send(application.store.revise(ident,obj))
                if action=='generate':return self.send(application.generate(ident,obj))
                if action=='results':return self.send(application.store.add_results(ident,obj.get('records',[])))
                if action=='questions':return self.send(application.store.add_question(ident,obj))
                raise ValueError('Unsupported operation')
            except Exception as exc:self.send({'error':str(exc),'details':getattr(exc,'details',None)},400)
        def do_POST(self):self.mutate('POST')
        def do_PATCH(self):self.mutate('PATCH')
    return Handler


def main():
    p=argparse.ArgumentParser(description='FlowPilotX general campaign studio')
    p.add_argument('--port',type=int,default=8788)
    p.add_argument('--storage')
    p.add_argument('--root',default=str(Path.cwd()),help='Study workspace; defaults to the current directory')
    p.add_argument('--profile',choices=['auto','demo','khu'],default='auto',
                   help='auto loads complete local KHU evidence or a synthetic demonstration when absent')
    args=p.parse_args()
    app=Application(args.root,args.storage,profile=args.profile)
    app.ensure_default()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(app))
    print(f'FlowPilotX http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
