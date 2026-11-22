"""Data-only component roles from explicitly recorded TRAIN native flags.

Flags are CAD annotation evidence, not a runtime perception input. No name
heuristics, simulator imports, pose reads, flag setters, or physics steps occur
here. Visual bindings always follow frozen TRAIN names; current flags cannot
silently admit an extra component.
"""
from collections.abc import Mapping
from numbers import Integral


SCHEMA = 'r10_TRAIN_native_component_roles_v1'
FLAG_NAMES = ('renderable', 'collidable', 'respondable')
# PyRep Object._get_property returns current & bit, despite a bool annotation.
# Source: backend/simConst.py. Respondable is an int32 parameter (0 or 1).
NATIVE_TRUE_VALUES = dict(renderable=512, collidable=1, respondable=1)


def _boolean(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and value in (0, 1):
        return bool(value)
    raise ValueError('explicit native boolean/0/1 component flag required: ' + name)


def component_roles(flags):
    """Preserve invisible physics and rendered non-collision components."""
    if not isinstance(flags, Mapping) or set(flags) != set(FLAG_NAMES):
        raise ValueError('record all three native component flags explicitly')
    raw = {name: _boolean(flags[name], name) for name in FLAG_NAMES}
    visual = raw['renderable']
    collision = raw['collidable'] or raw['respondable']
    category = ('both' if visual and collision else 'visual_only' if visual
                else 'collision_only' if collision else 'neither')
    return dict(schema=SCHEMA, flags=raw, visual=visual,
                physical_collision=collision, category=category)


def component_roles_from_native_values(raw):
    """Normalize exact PyRep getter types while retaining raw evidence."""
    if not isinstance(raw, Mapping) or set(raw) != set(FLAG_NAMES):
        raise ValueError('all raw native component getter values required')
    normalized = {}
    for name, value in raw.items():
        if isinstance(value, bool): normalized[name] = value
        elif isinstance(value, Integral) and value in (0, NATIVE_TRUE_VALUES[name]):
            normalized[name] = bool(value)
        else: raise ValueError('unexpected native special-property bit/int32 flag: ' + name)
    result = component_roles(normalized)
    result['native_getter_values'] = {name: value if isinstance(value, bool) else int(value)
                                    for name, value in raw.items()}
    result['native_true_values'] = NATIVE_TRUE_VALUES.copy()
    return result


def read_native_component_flags(shape):
    """TRAIN/audit caller owns permission; use only real getter API names."""
    return component_roles_from_native_values({name: getattr(shape, 'is_' + name)() for name in FLAG_NAMES})


def partition_components(records):
    """Classify a complete owned shape list without changing root ownership."""
    records = tuple(records)
    names = [row.get('name') if isinstance(row, Mapping) else None for row in records]
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError('nonempty unique owned component names required')
    classified = [{**dict(row), **component_roles(row.get('flags'))}
                  for row in sorted(records, key=lambda value: value['name'])]
    groups = {kind: [row['name'] for row in classified if row['category'] == kind]
              for kind in ('visual_only', 'collision_only', 'both', 'neither')}
    visual = [row['name'] for row in classified if row['visual']]
    collision = [row['name'] for row in classified if row['physical_collision']]
    return dict(schema=SCHEMA, components=classified, groups=groups,
                visual_component_names=visual, collision_component_names=collision,
                empty_visual_role=not visual, empty_collision_role=not collision,
                classification_uses_component_names=False,
                raw_special_flags_are_not_current_visibility_or_effective_model_collision=True)


def bind_frozen_visual_components(records, current_instance_ids):
    """Bind frozen visual names; never inspect mutable current native flags.

    The caller supplies the entire already-authorized current root ownership
    name table. Topology must match TRAIN, including excluded components.
    """
    partition = partition_components(records)
    if (not isinstance(current_instance_ids, Mapping)
            or set(current_instance_ids) != {row['name'] for row in partition['components']}):
        raise ValueError('current owned component names differ from frozen TRAIN topology')
    result = {}
    for name in partition['visual_component_names']:
        ids = current_instance_ids[name]
        if (not isinstance(ids, (tuple, list)) or not ids
                or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('current visual component requires unique nonnegative mask IDs')
        result[name] = tuple(int(value) for value in ids)
    return result
