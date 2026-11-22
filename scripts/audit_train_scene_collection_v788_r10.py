"""Reproduce one fixed TRAIN scene collection audit using stdlib CPU only.

This deliberately supports only the already-audited, SHA-bound task_design.ttt.
It decodes the existing CoppeliaSim 4.1 binary format, reads names/handles/parent
metadata and the Panda_arm collection, and skips all pose/mesh payloads. It does
not import PyRep, load a simulator, inspect TEST, or inspect live processes.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCENE = Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/task_design.ttt')
TRAIN_LOCK = ROOT/'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
OUTPUT = ROOT/'runtime/framework/train_scene_collection_static_v788.json'
SCENE_SHA = '11011817a890cecda1cf5e6b8ed9d7629998348a1ed74856ba65dfe6ce1257bd'
DECOMPRESSED_SHA = '0c0fe2e0e0e0b342f05a83e8c0b1c683ad64cd6cd97edbf8910fafcf447b2262'
COLLECTION_HEX = ('47726e0d0000000900000050616e64615f61726d4769780400000080841e00'
                  '50617201000000005569731400000010000000464439374335353333443436'
                  '43453935417367220000004f626a080000000c00000000000000477479010000'
                  '00015661720100000001454f4f454f4f')
REVIEW = ROOT/'third_party/coppeliaSimLib-v4.1.0-review/sourceCode'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_fixed_scene(raw):
    require(sha(raw) == SCENE_SHA, 'Only the exact frozen745 TRAIN scene is supported')
    require(raw[:4] == b'VREP', 'Unexpected file signature')
    version, minimum = struct.unpack_from('<ii', raw, 4)
    method, size = raw[12], struct.unpack_from('<I', raw, 13)[0]
    require((version, minimum, method, size) == (22, 18, 1, 7229018), 'Unexpected fixed scene header')
    # ser.cpp:609-670: 17-byte original header +1004 for serialization>12.
    source, position, nodes = raw[1021:], 0, 0

    def bit():
        nonlocal position
        require(position < len(source)*8, 'Truncated Huffman stream')
        value = (source[position >> 3] >> (7-(position & 7))) & 1
        position += 1
        return value

    def tree():
        nonlocal nodes
        nodes += 1
        require(nodes <= 511, 'Invalid Huffman tree')
        if bit():
            value = 0
            for _ in range(8):
                value = (value << 1) | bit()
            return value
        return tree(), tree()

    root = tree()
    decoded = bytearray(size)
    for index in range(size):
        node = root
        while isinstance(node, tuple):
            node = node[bit()]
        decoded[index] = node
    require(sha(decoded) == DECOMPRESSED_SHA, 'Decoded content differs from the already-audited scene')
    return decoded, dict(serialization_version=version, minimum_reader_version=minimum,
                        compression='Huffman', header_bytes=1021, decoded_bytes=size,
                        Huffman_tree_nodes=nodes)


def fields(data, terminal):
    """Read bounded three-letter metadata chunks, stopping at NXT or EOO."""
    position = 0
    while position < len(data):
        tag = bytes(data[position:position+3])
        require(len(tag) == 3, 'Truncated metadata tag')
        position += 3
        if tag == terminal:
            return
        require(position+4 <= len(data), 'Truncated metadata length')
        length = struct.unpack_from('<I', data, position)[0]
        position += 4
        require(position+length <= len(data), 'Metadata length exceeds its enclosing block')
        yield tag, data[position:position+length]
        position += length
    raise ValueError('Missing metadata terminal')


def string(value):
    require(len(value) >= 4, 'Truncated string length')
    length = struct.unpack_from('<I', value)[0]
    require(length+4 == len(value), 'String length differs')
    return bytes(value[4:]).decode('utf-8')


def collection(data, offset):
    result = dict(decoded_top_level_offset=offset, payload_bytes=len(data), payload_hex=bytes(data).hex(), elements=[])
    for tag, value in fields(data, b'EOO'):
        if tag == b'Grn':
            result['name'] = string(value)
        elif tag == b'Gix':
            require(len(value) == 4, 'Invalid collection handle')
            result['serialized_handle'] = struct.unpack('<i', value)[0]
        elif tag == b'Par':
            require(len(value) == 1, 'Invalid collection properties')
            result['overrideProperties'] = bool(value[0] & 1)
        elif tag == b'Asg':
            element = {}
            for key, part in fields(value, b'EOO'):
                if key == b'Obj':
                    require(len(part) == 8, 'Invalid collection element handles')
                    element['root_handle'], element['element_handle'] = struct.unpack('<ii', part)
                elif key == b'Gty':
                    require(len(part) == 1, 'Invalid collection element type')
                    element['type_value'] = part[0]
                elif key == b'Var':
                    require(len(part) == 1, 'Invalid collection element flags')
                    element['additive'] = bool(part[0] & 1)
            result['elements'].append(element)
    require(result['payload_hex'] == COLLECTION_HEX, 'Collection differs from the already-audited record')
    result['elements'][0]['type'] = 'fromBaseIncludingBase'
    return result


def parse_metadata(decoded):
    position, count, objects, collections = 0, 0, {}, []
    while position < len(decoded):
        offset = position
        tag = bytes(decoded[position:position+3])
        position += 3
        if tag == b'EOF':
            require(position == len(decoded), 'Trailing data after EOF')
            break
        require(position+4 <= len(decoded), 'Truncated top-level length')
        length = struct.unpack_from('<I', decoded, position)[0]
        position += 4
        require(position+length <= len(decoded), 'Top-level block exceeds file')
        data = decoded[position:position+length]
        position += length
        count += 1
        if data[:10] == b'3do\x03\x00\x00\x009:;':
            record = dict(type=tag.decode('ascii'))
            # Common scene-object fields end at NXT. Pose/mesh fields are skipped.
            for key, value in fields(data, b'NXT'):
                if key == b'Ids':
                    require(len(value) == 8, 'Invalid scene-object identity')
                    record['handle'], record['parent'] = struct.unpack('<ii', value)
                elif key == b'Nme':
                    record['name'] = string(value)
            require(set(record) == {'type', 'handle', 'parent', 'name'}, 'Missing object metadata')
            require(record['handle'] not in objects, 'Duplicate scene-object handle')
            objects[record['handle']] = record
        if tag == b'GRP':
            collections.append(collection(data, offset))
    require((count, len(objects), len(collections)) == (134, 79, 1), 'Unexpected fixed scene block inventory')
    require(decoded[-3:] == b'EOF', 'Missing final EOF')
    return objects, collections[0], count


def parent_chain(objects, handle):
    chain, seen = [], set()
    while handle in objects:
        require(handle not in seen, 'Parent cycle')
        seen.add(handle)
        record = objects[handle]
        chain.append(record)
        handle = record['parent']
    require(handle == -1, 'Unresolved initial parent')
    return chain


def source_evidence():
    entries = [
        (REVIEW/'serialization/ser.cpp', 609, 'Binary serialization>12 uses the 1021-byte header; lines712-717 decode Huffman'),
        (REVIEW/'serialization/huffman.c', 120, 'MSB-first bits; lines352-381 recover the tree; lines465-499 decode symbols'),
        (REVIEW/'collections/collection.cpp', 245, 'Grn/Gix/Par/Asg encode collection name/handle/overrideProperties/elements'),
        (REVIEW/'collections/collectionElement.cpp', 102, 'Obj/Gty/Var encode root and element handles/type/additive'),
        (REVIEW/'various/global.h', 265, 'Type1 is fromBaseIncludingBase'),
        (REVIEW/'sceneObjects/related/sceneObject.cpp', 1701, 'Ids contains handle/parent; Nme contains object name'),
        (REVIEW/'collections/collectionElement.cpp', 54, 'fromBaseIncludingBase recursively enumerates the current subtree'),
        (REVIEW/'sceneObjects/related/sceneObject.cpp', 3088, 'setParent notifies objectGotNewParent'),
        (REVIEW/'mainContainers/sceneContainers/sceneObjectContainer.cpp', 1203, 'Parent changes rebuild collection membership'),
        (REVIEW/'mainContainers/sceneContainers/collectionContainer.cpp', 121, 'With overrideProperties=False only effectively collidable members enter the collision group'),
        (ROOT/'our_mani/multitask/maps.py', 261, 'public_scene temporarily disables collidable on initial native nonrobot shapes, including graspable bodies'),
    ]
    author = SCENE.parents[3]
    entries.extend([
        (author/'third_party/RLBench/rlbench/backend/const.py', 27, 'TTT_FILE selects task_design.ttt'),
        (author/'third_party/RLBench/rlbench/environment.py', 95, 'Panda uses the robot already present in task_design.ttt'),
        (author/'third_party/PyRep/pyrep/robots/end_effectors/gripper.py', 50, 'grasp reparents the grasped body to the gripper attachPoint'),
    ])
    return [dict(path=str(path), sha256=sha(path.read_bytes()), line=line, basis=basis)
            for path, line, basis in entries]


def main():
    raw, lock_raw = SCENE.read_bytes(), TRAIN_LOCK.read_bytes()
    lock = json.loads(lock_raw)
    require(lock.get('files', {}).get(str(SCENE)) == SCENE_SHA, 'Exact745 TRAIN scene binding absent')
    decoded, header = decode_fixed_scene(raw)
    objects, group, count = parse_metadata(decoded)
    require(group['name'] == 'Panda_arm' and group['overrideProperties'] is False, 'Unexpected fixed collection')
    root = group['elements'][0]['root_handle']
    require(root == 12 and objects[root]['name'] == 'Panda', 'Unexpected Panda root')
    members = [row for row in objects.values() if any(part['handle'] == root for part in parent_chain(objects, row['handle']))]
    require(len(members) == 42, 'Unexpected initial Panda subtree')
    by_name = {row['name']: row for row in members}
    names = ('Panda_gripper', 'Panda_leftfinger_respondable', 'Panda_rightfinger_respondable',
             'Panda_leftfinger_force_contact', 'Panda_rightfinger_force_contact', 'Panda_gripper_attachPoint')
    chains = {name: parent_chain(objects, by_name[name]['handle']) for name in names}
    result = dict(schema='r10_fixed_TRAIN_scene_collection_static_audit_v788',
        created_utc=datetime.now(timezone.utc).isoformat(), complete=True, CPU_only=True,
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).read_bytes())),
        source_scene=dict(path=str(SCENE), sha256=sha(raw)),
        frozen_TRAIN_binding=dict(path=str(TRAIN_LOCK.relative_to(ROOT)), sha256=sha(lock_raw), scene_sha256=SCENE_SHA),
        decoded_sha256=sha(decoded), header=header, top_level_chunks=count, scene_objects=len(objects),
        collection=group, root=objects[root], initial_structural_member_count=len(members),
        initial_structural_members=members, gripper_parent_chains=chains, source_evidence=source_evidence(),
        conclusions=dict(initial_collection_includes_gripper_subtree=True,
            grasp_reparent_adds_body_to_structural_collection_under_these_rules=True,
            initial_overrideProperties=False,
            disabled_native_grasped_shape_filtered_from_effective_collision_group_under_these_rules=True,
            native_grasped_shape_remains_in_group_due_to_override_hypothesis_supported=False,
            establishes_actual_collision_pair=False, establishes_lift_rejection_root_cause=False),
        scope=dict(frozen_initial_scene_and_static_sources_only=True, live_collection_inventory_read=False,
            runtime_collection_property_mutation_excluded_by_this_audit=False, runtime_member_flags_read=False,
            effective_runtime_collision_members_enumerated=False, native_simulator_started=False,
            native_imported=False, raw_CFFI_or_Lua_called=False, TEST_read=False,
            target_poses_output=False, pose_or_mesh_payloads_interpreted=False,
            current_VAL_TEST_jobs_or_frozen_files_modified=False))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=OUTPUT.parent, prefix=OUTPUT.name+'.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, OUTPUT)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(dict(complete=True, CPU_only=True, output=str(OUTPUT.relative_to(ROOT)),
        output_sha256=sha(OUTPUT.read_bytes()), script_sha256=result['script']['sha256'],
        scene_sha256=sha(raw), decoded_sha256=sha(decoded), collection=group['name'],
        overrideProperties=False, initial_structural_members=len(members),
        native_started=False, TEST_read=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
