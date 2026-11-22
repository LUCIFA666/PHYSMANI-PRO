"""Public language plus current RGB-D colours resolve task target roles.

The semantic instance/component bindings and colour vocabulary are a declared
shared prior. They are not described as a detector learned from raw RGB-D.
Native variation IDs, target_spoke, teacher actions and goal sensors are not
inputs. TRAIN teacher colour annotations may only supervise the offline fit.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

COLOUR_SCHEMA = 'r10_TRAIN_public_colour_component_prototypes_v1'


class PublicRoleResolutionError(ValueError):
    pass


def _sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _normalise_words(text):
    return re.sub(r'\s+', ' ', re.sub(r'[-_]', ' ', str(text).lower())).strip()


def _lab(rgb):
    """Fixed sRGB/D65 CIELAB descriptor; no learned or native colour lookup."""
    rgb = np.asarray(rgb, float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not len(rgb) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 255)):
        raise PublicRoleResolutionError('current/ TRAIN colour evidence must be finite RGB255 pixels')
    srgb = rgb/255.
    linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
    xyz = linear @ np.array([[.4124564,.3575761,.1804375],
                             [.2126729,.7151522,.0721750],
                             [.0193339,.1191920,.9503041]]).T
    xyz /= np.array([.95047, 1., 1.08883])
    delta = 6/29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    return np.c_[116*f[:,1]-16, 500*(f[:,0]-f[:,1]), 200*(f[:,1]-f[:,2])]/np.array([100.,128.,128.])


@dataclass(frozen=True)
class TRAINColourSample:
    task_name: str
    source_group: str
    role_id: str
    component_names: tuple[str, ...]
    colour_name: str
    rgb_pixels_255: np.ndarray
    aliases: tuple[str, ...] = ()
    split: str = 'train'

    def __post_init__(self):
        if (self.split != 'train' or not self.source_group.startswith('train:'+self.task_name+':')
                or not self.role_id or not self.component_names
                or len(set(self.component_names)) != len(self.component_names)
                or not _normalise_words(self.colour_name)):
            raise PublicRoleResolutionError('colour fitting requires explicit TRAIN source/role/components')
        _lab(self.rgb_pixels_255)
        object.__setattr__(self, 'colour_name', _normalise_words(self.colour_name))


@dataclass(frozen=True)
class ColourFitConfig:
    minimum_sources_per_colour: int
    within_colour_quantile: float
    minimum_acceptance_radius: float
    minimum_margin: float
    minimum_visible_pixels: int
    maximum_observation_age: float

    def __post_init__(self):
        if (self.minimum_sources_per_colour < 2 or self.minimum_visible_pixels < 1
                or not np.isfinite(list(asdict(self).values())).all()
                or not 0 < self.within_colour_quantile <= 1
                or min(self.minimum_acceptance_radius, self.minimum_margin, self.maximum_observation_age) <= 0):
            raise PublicRoleResolutionError('explicit TRAIN colour fit and observation thresholds required')


def fit_colour_artifact(samples, config, *, TRAIN_manifest_path=None, synthetic=False):
    """Source-balanced colour fit; generated replicas stay in their raw group."""
    samples = tuple(samples)
    if not samples:
        raise PublicRoleResolutionError('missing TRAIN colour supervision')
    groups = sorted({sample.source_group for sample in samples})
    manifest_sha = None
    if not synthetic:
        if TRAIN_manifest_path is None:
            raise PublicRoleResolutionError('production colour fit requires a real TRAIN manifest')
        path = Path(TRAIN_manifest_path)
        entries = json.loads(path.read_text()).get('entries', [])
        if not entries or any(row.get('split') != 'train' for row in entries):
            raise PublicRoleResolutionError('colour calibration manifest must be TRAIN-only')
        declared = {':'.join(('train', row['task'], row.get('parent_raw_episode_id') or row['raw_episode_id']))
                    for row in entries}
        if set(groups)-declared:
            raise PublicRoleResolutionError('colour labels contain undeclared TRAIN parent sources')
        manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    component_map, labels = {}, {}
    for sample in samples:
        key = sample.task_name+'|'+sample.role_id
        components = tuple(sorted(sample.component_names))
        if key in component_map and component_map[key] != components:
            raise PublicRoleResolutionError('colour-bearing component identity changes across TRAIN sources')
        component_map[key] = components
        labels.setdefault(sample.colour_name, []).append(sample)
    prototypes, names = {}, sorted(labels)
    if len(names) < 2:
        raise PublicRoleResolutionError('at least two TRAIN colour classes are required for ambiguity checking')
    for name, rows in labels.items():
        source_groups = sorted({sample.source_group for sample in rows})
        if len(source_groups) < config.minimum_sources_per_colour:
            raise PublicRoleResolutionError('insufficient distinct TRAIN sources for colour '+name)
        centres = [np.median([np.median(_lab(sample.rgb_pixels_255), axis=0)
                             for sample in rows if sample.source_group == group], axis=0)
                   for group in source_groups]
        prototype = np.median(centres, axis=0)
        radii = np.linalg.norm(np.asarray(centres)-prototype, axis=1)
        aliases = {_normalise_words(alias) for sample in rows for alias in sample.aliases}
        prototypes[name] = dict(prototype_lab_normalised=prototype.tolist(),
            acceptance_radius=max(config.minimum_acceptance_radius,
                float(np.quantile(radii, config.within_colour_quantile))),
            aliases=sorted(aliases | {name}), TRAIN_parent_source_groups=source_groups)
    used_aliases = [alias for prototype in prototypes.values() for alias in prototype['aliases']]
    if len(used_aliases) != len(set(used_aliases)):
        raise PublicRoleResolutionError('ambiguous TRAIN colour aliases')
    for name in names:
        nearest = min(np.linalg.norm(np.array(prototypes[name]['prototype_lab_normalised'])-
                      np.array(prototypes[other]['prototype_lab_normalised'])) for other in names if other != name)
        if nearest <= 2*config.minimum_margin:
            raise PublicRoleResolutionError('TRAIN colour prototypes are not separable at the frozen margin')
        prototypes[name]['acceptance_radius'] = min(prototypes[name]['acceptance_radius'], nearest/2)
    payload = dict(metadata=dict(schema=COLOUR_SCHEMA, fit_complete=True, fit_split='train',
        TEST_used_for_fitting=False, synthetic=bool(synthetic),
        TRAIN_parent_source_groups=groups, TRAIN_manifest_path=str(TRAIN_manifest_path) if TRAIN_manifest_path else None,
        TRAIN_manifest_sha256=manifest_sha, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_semantic_binding_is_extra_prior=True, pure_RGBD_detector_claimed=False,
        fit_configuration=asdict(config), colour_observations_source='current_RGBD_role_component_pixels',
        teacher_labels_used_only_in_TRAIN=True),
        role_components={key:list(value) for key,value in component_map.items()}, colours=prototypes)
    payload['artifact_sha256'] = _sha(payload)
    return payload


class PublicRoleResolver:
    def __init__(self, task_spec, colour_artifact, *, shared_semantic_permission, allow_synthetic=False):
        self.spec, self.artifact = task_spec, colour_artifact
        m = colour_artifact.get('metadata', {})
        if (shared_semantic_permission is not True or m.get('schema') != COLOUR_SCHEMA
                or m.get('fit_complete') is not True or m.get('fit_split') != 'train'
                or m.get('TEST_used_for_fitting') is not False
                or (m.get('synthetic') is not False and not allow_synthetic)
                or not m.get('TRAIN_parent_source_groups')
                or _sha({k:v for k,v in colour_artifact.items() if k != 'artifact_sha256'}) != colour_artifact.get('artifact_sha256')
                or m.get('source_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise PublicRoleResolutionError('frozen TRAIN colour artifact and common semantic-ID permission required')
        if m.get('synthetic') is False and hashlib.sha256(Path(m['TRAIN_manifest_path']).read_bytes()).hexdigest() != m['TRAIN_manifest_sha256']:
            raise PublicRoleResolutionError('TRAIN colour source manifest changed')
        self.config = ColourFitConfig(**m['fit_configuration'])

    def _requested_colour(self, descriptions):
        occurrences = []
        noun = r'(?:spoke|peg)' if self.spec.family == 'insert_onto_rotating_peg' else r'(?:button|base)'
        for description_index, text in enumerate(descriptions):
            text = _normalise_words(text)
            for name, colour in self.artifact['colours'].items():
                for alias in colour['aliases']:
                    # The author also emits "movingred button" without a space.
                    expression = r'(?<![a-z])(?:moving\s*)?'+re.escape(alias)+r'\s+(?:colou?red\s+)?'+noun+r'\b'
                    occurrences.extend((description_index, match.start(), match.end(), name)
                        for match in re.finditer(expression, text))
        # "light blue spoke" must not also request the shorter "blue spoke".
        requested = {name for d,start,end,name in occurrences if not any(
            d == other_d and other_start <= start and end <= other_end
            and (other_start < start or end < other_end)
            for other_d,other_start,other_end,_ in occurrences)}
        if len(requested) != 1:
            raise PublicRoleResolutionError('public descriptions do not specify one known target colour')
        return next(iter(requested))

    def _role_colour(self, role, observation, template):
        key = self.spec.task_name+'|'+role
        components = self.artifact['role_components'].get(key)
        if not components:
            raise PublicRoleResolutionError('TRAIN artifact lacks colour component supervision for '+role)
        ids = {identity for component in template.components if component.name in components
               for identity in component.instance_ids}
        if {component.name for component in template.components if component.name in components} != set(components):
            raise PublicRoleResolutionError('current component topology differs from TRAIN colour prior')
        values = []
        images, masks = observation['observation/images'], observation['observation/instance_masks']
        for camera in sorted(set(images) & set(masks)):
            mask, rgb = np.asarray(masks[camera]), np.asarray(images[camera])
            if mask.ndim == 3 and mask.shape[2] == 3:
                channels = np.rint(mask*255.).astype(np.int64)
                mask = channels[...,0]+256*channels[...,1]+65536*channels[...,2]
            if rgb.shape[:2] != mask.shape or rgb.shape[-1] != 3:
                raise PublicRoleResolutionError('misaligned current RGB/instance colour pixels')
            pixels = rgb[np.isin(mask, tuple(ids))]
            if len(pixels):
                values.append(pixels)
        count = sum(len(value) for value in values)
        if count < self.config.minimum_visible_pixels:
            return dict(valid=False, reason='insufficient_current_visible_colour_pixels', pixels=count)
        descriptor = np.median(_lab(np.concatenate(values)), axis=0)
        distances = sorted((float(np.linalg.norm(descriptor-np.array(colour['prototype_lab_normalised']))), name)
                           for name,colour in self.artifact['colours'].items())
        distance, name = distances[0]
        margin = distances[1][0]-distance
        return dict(valid=bool(distance <= self.artifact['colours'][name]['acceptance_radius']
                              and margin >= self.config.minimum_margin),
            colour_name=name, distance=distance, margin=margin, pixels=count,
            provenance='current_RGBD_pixels_with_shared_semantic_instance_component_bindings',
            colour_artifact_sha256=self.artifact['artifact_sha256'])

    def resolve(self, descriptions, state, observation, templates):
        """Return public observed role bindings, or raise without a fallback guess."""
        if self.spec.family not in ('insert_onto_rotating_peg', 'push_moving_button'):
            raise PublicRoleResolutionError('this resolver only supplies publicly colour-specified target roles')
        if isinstance(descriptions, str):
            descriptions = (descriptions,)
        if not descriptions or any(not isinstance(text, str) for text in descriptions):
            raise PublicRoleResolutionError('public task language is required')
        now = float(state.robot.timestamp)
        if abs(float(observation['timestamp'])-now) > 1e-8:
            raise PublicRoleResolutionError('target language resolution requires current RGBD/proprioception')
        target = self._requested_colour(tuple(descriptions))
        template_by_role = {template.role_id:template for template in templates}
        roles = ([role.name for role in self.spec.roles if re.fullmatch(r'peg[0-9]+',role.name)]
                 if self.spec.family == 'insert_onto_rotating_peg' else ['button'])
        evidence, matches = {}, []
        for role in roles:
            body = state.bodies.get(role)
            meta = state.perception_metadata.get('bodies', {}).get(role,{})
            if (body is None or not body.visible or meta.get('directly_visible') is not True
                    or meta.get('registration_valid') is not True or role not in template_by_role
                    or now-body.timestamp < -1e-8 or now-body.timestamp > self.config.maximum_observation_age):
                evidence[role] = dict(valid=False, reason='role_not_fresh_directly_observed')
                continue
            evidence[role] = self._role_colour(role, observation, template_by_role[role])
            if evidence[role]['valid'] and evidence[role]['colour_name'] == target:
                matches.append(role)
        if len(matches) != 1:
            raise PublicRoleResolutionError('public target colour is absent or ambiguous among current observed roles')
        selected_role = 'selected_peg' if self.spec.family == 'insert_onto_rotating_peg' else 'selected_button'
        bindings = {**state.role_bindings, selected_role:(matches[0],)}
        if selected_role == 'selected_button':
            bindings['button'] = (matches[0],)
        return bindings, dict(public_target_colour=target, selected_role=selected_role, body_id=matches[0],
            role_colour_evidence=evidence, semantic_instance_bindings_are_extra_shared_prior=True,
            pure_RGBD_detector_claimed=False, native_target_or_variation_read=False,
            teacher_labels_used_at_inference=False, colour_artifact_sha256=self.artifact['artifact_sha256'])
