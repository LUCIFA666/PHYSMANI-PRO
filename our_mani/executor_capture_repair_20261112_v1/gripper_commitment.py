"""Complete accepted release events across short model intent pulses."""
class GripperCommitment:
    def __init__(self):self.release_latched=False

    def resolve(self,requested,*,actual_target,done,held):
        if requested not in (0.,1.):raise ValueError('Binary intent required')
        if self.release_latched and actual_target==1. and done is True:
            self.release_latched=False
        if self.release_latched:return 1.
        if requested==1. and held:
            self.release_latched=True
            return 1.
        if done is False and requested!=actual_target:return None
        return requested

    def brake(self,*,actual_target,done):
        # Cancel a release that has not started after pose rejection. A real
        # in-flight opening still completes through the unchanged core driver.
        if not(actual_target==1. and done is False):self.release_latched=False
