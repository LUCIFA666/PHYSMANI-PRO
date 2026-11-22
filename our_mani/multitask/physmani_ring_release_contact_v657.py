"""Keep the original insertion contact role during the author's wp4-open.

wp4 and wp4-open track the same selected peg pose. Opening the gripper does
not turn that peg into a forbidden robot contact between those two stages.
All other arm/world and held-ring/world collision checks remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
from .evaluation import file_sha
from .physmani_terminal_opening_v620 import AuthorRingReleaseIntents as Original
from .physmani_ring_release_v590 import binding as release_binding


def binding(root,native_repo,task):
    source_rule=release_binding(root,native_repo,task)
    return dict(schema='r10_author_wp4_open_retains_selected_peg_insertion_contact_v657',task=task,
        source_author_terminal_rule=source_rule,semantic_phase='release',collision_contact_phase='insert',
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_ring_release_contact_v657.py'),
        same_selected_peg_only=True,held_ring_vs_world_and_other_obstacles_still_checked=True,
        physical_gripper_gate_native_RML_and_success_unchanged=True)


class SourceContactRingIntents(Original):
    def __init__(self,*a,contact_rule,**kw):
        super().__init__(*a,**kw)
        if (contact_rule.get('schema')!='r10_author_wp4_open_retains_selected_peg_insertion_contact_v657'
            or contact_rule.get('task')!=self.task or contact_rule.get('collision_contact_phase')!='insert'
            or contact_rule.get('held_ring_vs_world_and_other_obstacles_still_checked') is not True):
            raise ValueError('same selected insertion contact must persist through source wp4-open')
        self.contact_rule=deepcopy(contact_rule);self.contact_events=[]

    def admit_joint(self,joints):
        original=self.selected
        if original is None or original['phase']!='release':return super().admit_joint(joints)
        if original['template'].phase!='insert':raise ValueError('only source wp4 continuation can retain insertion contact')
        self.selected=dict(original,phase='insert')
        try:
            accepted,proof=super().admit_joint(joints)
            proof=dict(proof,semantic_phase='release',collision_contact_phase='insert',
                original_selected_insertion_contact_retained=True,other_collision_checks_unchanged=True)
            self.contact_events.append(dict(time=float(self.frontend.state.robot.timestamp),accepted=accepted,proof=deepcopy(proof)))
            return accepted,proof
        finally:self.selected=original

    def receipt(self):return dict(super().receipt(),source_wp4_open_contact=dict(
        binding=deepcopy(self.contact_rule),events=deepcopy(self.contact_events)))
