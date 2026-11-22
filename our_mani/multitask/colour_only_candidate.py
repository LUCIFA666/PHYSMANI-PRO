"""Independent TRAIN colour experiment; no runtime semantic admission.

The original fitting and crossfit kernels are called unchanged.  The additional
visual patch is a colour descriptor, never a native instance/component label.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .artifact_fitting import colour_parent_crossfit, colour_samples
from .public_roles import ColourFitConfig, TRAINColourSample, fit_colour_artifact, PublicRoleResolutionError

SCHEMA = 'r10_development_TRAIN_colour_only_visual_supplement_candidate_v1'
VISUAL_ROLE = '__visual_colour_prototype_patch_only__'
VISUAL_COMPONENT = '__no_runtime_component_authority__'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pointer(root, path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(root)), sha256=sha(path))


def bound(root, ref):
    path = (root / ref['path']).resolve()
    require(path.is_relative_to(root) and sha(path) == ref['sha256'], 'bound input changed: '+str(path))
    return path


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def verify_native_colour_cases(root, proof_path):
    """Verify all canonical/raw JSON and colour bytes; reuse prior motion audit."""
    proof = read(proof_path)
    require(proof.get('audit_passed') is True and proof.get('complete') is True,
            'actual complete TRAIN bundle proof required')
    projection = proof['acquisition_projection']
    require(projection['split'] == 'train' and projection['TEST_used'] is False
            and projection['complete'] is True and projection['completed'] == 3100
            and not projection['missing_case_ids'], 'exact complete 3100 TRAIN projection required')
    entries = {(row['task'], row['raw_episode_id']): row for row in projection['entries']}
    require(len(entries) == len(projection['entries']) == len(projection['case_receipts']) == 3100,
            'duplicate/missing native acquisition')
    verified = proof['verified_files_sha256']
    cases, ledger, seen, parents = [], [], set(), set()
    for item in projection['case_receipts']:
        path = bound(root, item)
        require(verified.get(str(path)) == item['sha256'], 'canonical not in actual3100 proof')
        record = read(path)
        key = (record['task'], record['raw_episode_id'])
        require(key in entries and key not in seen, 'native case identity duplicate/unknown')
        seen.add(key)
        entry = entries[key]
        group = 'train:'+entry['task']+':'+entry['parent_raw_episode_id']
        require(record['status'] == 'completed' and record['split'] == entry['split'] == 'train'
                and record['case_id'] == item['case_id'] and record['entry'] == entry
                and record['parent_source_group'] == entry['parent_source_group'] == group
                and record['plan_sha256'] == item['actual_plan_sha256']
                == proof['per_case_actual_plan_sha256'][item['case_id']], 'native identity/projection mismatch')
        parent = record['parent_process_receipt']
        raw_ref = dict(path=parent['child_case_path'], sha256=parent['child_case_sha256'])
        raw = bound(root, raw_ref)
        require(verified.get(str(raw)) == raw_ref['sha256'], 'raw child not in actual3100 proof')
        require(read(raw) == {k:v for k,v in record.items() if k != 'parent_process_receipt'},
                'canonical differs from original raw child')
        names = [name for name in record['files_sha256'] if Path(name).name == 'TRAIN_color_annotations.json']
        require(len(names) == 1, 'exact one native colour sidecar required')
        ref = dict(path=names[0], sha256=record['files_sha256'][names[0]])
        sidecar = bound(root, ref)
        require(verified.get(str(sidecar)) == ref['sha256'], 'colour sidecar not in actual3100 proof')
        parents.add(group)
        cases.append(dict(path=path, record=record, entry=entry))
        ledger.append(dict(case_id=item['case_id'], parent_source_group=group,
            actual_plan_sha256=record['plan_sha256'], canonical=pointer(root,path),
            raw_child=pointer(root,raw), colour_sidecar=pointer(root,sidecar)))
    require(len(parents) == 1400, 'original1400 parent roster changed')
    return cases, ledger, sorted(parents)


def load_visual_patch(root, annotation_path, original_parents, peer_path):
    annotation = read(annotation_path)
    peer = read(peer_path)
    require(peer.get('annotation') == pointer(root, annotation_path), 'visual peer annotation binding differs')
    require(annotation['split'] == 'train' and annotation['frame'] == 0
        and annotation['VAL_or_TEST_used'] is False
        and annotation['training_admitted'] is False
        and annotation['native_teacher_colour_annotation_available'] is False
        and annotation['label_inferred_using_existing_classifier'] is False
        and annotation['no_runtime_semantic_permission_granted'] is True
        and 'not original author/native teacher metadata' in annotation['annotation_source'],
        'visual annotation provenance/admission differs')
    group = 'train:'+annotation['task']+':'+annotation['raw_episode_id']
    require(group == annotation['parent_source_group'] and group not in original_parents,
            'visual patch is not an independent TRAIN parent')
    roster = read(bound(root, annotation['all200_roster']))
    selection = read(bound(root, annotation['predeclared_fixed16_selection']))
    official = read(bound(root, roster['source_manifest']))
    excluded = {'pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed'}
    rows = sorted([row for row in official['episodes'] if row['task'] in excluded],
                  key=lambda row:(row['task'],row['raw_episode_id']))
    fixed16 = [row for task in sorted(excluded) for row in [r for r in rows if r['task']==task][:8]]
    require(official['split'] == roster['split'] == 'train' and len(rows) == 200
        and rows == roster['rows'] and selection['selected_parents'] == fixed16
        and any(row['task']==annotation['task'] and row['raw_episode_id']==annotation['raw_episode_id']
                for row in fixed16), 'visual parent not in fixed16/full200 official TRAIN roster')
    exported = read(bound(root, annotation['source_export_manifest']))
    exported_row = next(row for row in exported['images'] if row['PNG']==annotation['source_image'])
    rgb = np.asarray(Image.open(bound(root, annotation['source_image'])))
    require(rgb.dtype == np.uint8 and rgb.shape == (256,256,3)
        and hashlib.sha256(rgb.tobytes()).hexdigest() == annotation['source_RGB_pixel_array_sha256']
        == exported_row['original_pixel_sha256'] == exported_row['decoded_PNG_pixel_sha256'],
        'visual RGB pixels differ from independently verified frame0')
    mask = np.asarray(Image.open(bound(root, annotation['patch_mask'])))
    recreated = Image.new('L',(256,256),0)
    for polygon in annotation['polygons_xy']:
        ImageDraw.Draw(recreated).polygon([tuple(p) for p in polygon],fill=255)
    pixels = np.load(bound(root, annotation['RGB_pixels']),allow_pickle=False)
    require(mask.dtype == np.uint8 and np.array_equal(mask,np.asarray(recreated))
        and pixels.dtype == np.uint8 and pixels.shape == (80,3)
        and np.array_equal(pixels,rgb[mask>0]) and annotation['visible_pixels']==80,
        'visual patch mask/80 original pixels changed')
    # These two sentinel fields satisfy only the unchanged numerical kernel's
    # bookkeeping. They are stripped from the candidate's native role map.
    sample = TRAINColourSample(annotation['task'],group,VISUAL_ROLE,(VISUAL_COMPONENT,),
                              annotation['colour_label'],pixels)
    return sample, annotation


def confusion_and_parent_results(crossfit, samples):
    rows = crossfit['held_predictions']
    groups = sorted({sample.source_group for sample in samples})
    by_parent = {group:[row for row in rows if row['source_group']==group] for group in groups}
    parent_results = [dict(source_group=group, fold=crossfit['parent_to_fold'][group],
        held_colour_samples=len(by_parent[group]),
        correct_and_supported_fraction=(float(np.mean([r['correct_and_supported'] for r in by_parent[group]]))
                                        if by_parent[group] else None),
        failures=[r for r in by_parent[group] if not r['correct_and_supported']]) for group in groups]
    confusion = defaultdict(Counter)
    for row in rows:
        confusion[row['true_public_colour']][row['predicted_colour']]+=1
    return parent_results,{name:dict(counts) for name,counts in sorted(confusion.items())}


def run(root, output, *, proof_path, annotation_path, peer_path, config_path):
    root,output = Path(root).resolve(),Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'fresh ROOT output namespace required')
    output.mkdir(parents=True)
    sources = [Path(__file__), root/'our_mani/multitask/artifact_fitting.py',
        root/'our_mani/multitask/public_roles.py',root/'our_mani/multitask/evaluation.py',
        root/'scripts/fit_colour_only_visual_candidate_r10.py',root/'tests/test_colour_only_visual_candidate_r10.py']
    bindings = [pointer(root,path) for path in sources+[proof_path,annotation_path,peer_path,config_path]]
    for ref in bindings:
        path=bound(root,ref)
        destination=output/'sources'/ref['sha256']/path.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write(output/'source_bindings.json',dict(before=bindings,unchanged_original_kernels=True))
    started = datetime.now(timezone.utc).isoformat()
    cases,ledger,parents = verify_native_colour_cases(root,proof_path)
    write(output/'actual_colour_input_ledger.json',dict(schema='r10_colour_only_actual_input_ledger_v1',
        native_acquisitions=3100,native_parent_sources=1400,entries=ledger,
        reused_actual3100_motion_and_lineage_audit=pointer(root,proof_path),
        canonical_raw_and_colour_bytes_rehashed=True,unrelated_motion_arrays_rehashed=False))
    configuration = read(config_path)
    config = ColourFitConfig(**configuration['colour'])
    require(asdict(config) == dict(minimum_sources_per_colour=8,within_colour_quantile=.95,
        minimum_acceptance_radius=.05,minimum_margin=.02,minimum_visible_pixels=20,
        maximum_observation_age=.1) and configuration['colour_parent_cv_minimum_accuracy']==.95,
        'original frozen colour numerical parameters changed')
    native,excluded = colour_samples(root,cases,config)
    visual,annotation = load_visual_patch(root,annotation_path,parents,peer_path)
    samples = native+[visual]
    require(len({s.source_group for s in native})==400 and len({s.source_group for s in samples})==401,
            'actual colour parent counts differ')
    manifest = dict(schema='r10_TRAIN_colour_only_native3100_plus_visual_one_parent_v1',
        split='train',VAL_used=False,TEST_used=False,native_acquisition_count=3100,
        native_parent_count=1400,native_colour_parent_count=400,visual_colour_parent_count=1,
        Context_K1_parent_count_unchanged=1400,interaction_acquisitions_unchanged=3100,
        entries=[dict(split='train',task=row['task'],raw_episode_id=row['raw_episode_id'],
            parent_raw_episode_id=row['parent_raw_episode_id']) for row in (case['entry'] for case in cases)]
            +[dict(split='train',task=annotation['task'],raw_episode_id=annotation['raw_episode_id'],
                source_kind='additional_Codex_visual_colour_patch_only',annotation=pointer(root,annotation_path))],
        visual_label_origin=annotation['annotation_source'],no_new_native_teacher_acquisition=True,
        no_visual_instance_component_semantic_authority=True)
    write(output/'colour_only_manifest.json',manifest)
    crossfit=colour_parent_crossfit(samples,config,minimum_accuracy=.95)
    write(output/'TRAIN_parent_crossfit.json',crossfit)
    parent_results,confusion=confusion_and_parent_results(crossfit,samples)
    write(output/'all_heldout_parent_results.json',parent_results)
    write(output/'heldout_confusion.json',confusion)
    raw,error=None,None
    try:
        raw=fit_colour_artifact(samples,config,TRAIN_manifest_path=output/'colour_only_manifest.json')
    except PublicRoleResolutionError as caught:
        error=str(caught)
    native_map={s.task_name+'|'+s.role_id:sorted(s.component_names) for s in native}
    sentinel=visual.task_name+'|'+VISUAL_ROLE
    if raw is not None:
        require(raw['role_components'].pop(sentinel)==[VISUAL_COMPONENT]
            and raw['role_components']==native_map,'visual sample changed native component map')
    candidate=dict(metadata=dict(schema=SCHEMA,development_candidate=True,formal_release=False,
        fit_split='train',VAL_used=False,TEST_used=False,TRAIN_parent_crossfit=crossfit,
        fit_configuration=asdict(config),original_numerical_kernel_sources=bindings[:4],
        manifest=pointer(root,output/'colour_only_manifest.json'),visual_annotation=pointer(root,annotation_path),
        visual_label_origin=annotation['annotation_source'],runtime_component_permission_granted=False,
        original_native_role_map_unchanged=True,prototype_crossfit_passed=crossfit['passed'],
        full_colour_kernel_fit_complete=raw is not None,full_colour_kernel_rejection=error),
        native_role_components=native_map,colours=(raw['colours'] if raw is not None else None))
    write(output/'candidate.json',candidate)
    for ref in bindings: bound(root,ref)
    # The sidecars and raw/canonical JSON are rechecked after fitting too.
    for row in ledger:
        for key in ('canonical','raw_child','colour_sidecar'): bound(root,row[key])
    report=dict(schema=SCHEMA,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
        actual_fit_complete=raw is not None,full_colour_kernel_rejection=error,development_only=True,formal_release=False,
        native_acquisitions_unchanged=3100,native_parent_sources_unchanged=1400,
        native_colour_parent_sources=400,additional_visual_colour_parent_sources=1,
        total_colour_parent_sources=401,native_samples=len(native),visual_samples=1,
        excluded_native_visible_samples=excluded,colour_classes=len(crossfit['public_colour_class_parent_support']),
        original_greedy_crossfit_used=True,alternate_assignment_used=False,
        fold_class_support={str(row['fold']):row['fit_parent_support_by_colour'] for row in crossfit['folds']},
        crossfit_valid=crossfit['valid'],crossfit_passed=crossfit['passed'],
        macro_heldout_parent_correct_and_supported_fraction=crossfit['macro_parent_correct_and_supported_fraction'],
        required_fraction=.95,all_original_colour_data_retained=True,no_accuracy_selected_folds=True,
        original_component_map_unchanged=True,source_bindings_before_after_identical=True,
        actual_input_hashes_verified_before_after=True,motion_array_audit_reused_not_recomputed=True,
        no_native_GPU_VAL_TEST_or_runtime_semantic_admission=True,
        outputs={name:pointer(root,output/name) for name in ('candidate.json','colour_only_manifest.json',
            'TRAIN_parent_crossfit.json','all_heldout_parent_results.json','heldout_confusion.json',
            'actual_colour_input_ledger.json','source_bindings.json')})
    write(output/'report.json',report)
    return report
