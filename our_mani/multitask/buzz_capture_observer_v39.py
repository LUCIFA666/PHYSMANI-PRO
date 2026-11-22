"""TRAIN metric-only capture of buffers already supplied to an unchanged actor."""
from pathlib import Path
from types import MethodType
from .buzz_visual_capture import CandidateVisualCaptureWriter,NAMESPACE,COMPONENTS


class BuzzCaptureObserver:
    def __init__(self,actor,prepared,env):
        args=prepared['args'];task=args.task
        components={component.name:list(component.instance_ids) for template in actor.tracker.templates.values()
            for component in template.components if component.name in COMPONENTS}
        directory=NAMESPACE/('native_v39_'+task+'_'+args.raw_episode_id+'_'+args.method)
        identity=dict(split='train',task=task,parent_source_group='train:'+task+':'+args.raw_episode_id,
            source_raw_episode_id=args.raw_episode_id,replica=0,seed=int(env.seed),variation=int(env.reset_receipt['variation']))
        self.writer=CandidateVisualCaptureWriter(directory,identity=identity,component_instance_ids=components,
            canonical_CAD=prepared['config']['learned_artifacts'][task]['geometry'])
        self.finished=None;previous_act=actor.act
        def act(instance,observation,robot,remaining_seconds):
            tick=round(float(robot.timestamp)/.05)
            if tick<5 or tick%10==0:self.writer.append(observation,robot)
            return previous_act(observation,robot,remaining_seconds)
        actor.act=MethodType(act,actor)
        previous_metrics=actor.metrics_report
        def metrics(instance):
            if self.finished is None:self.finished=self.writer.finish()
            return dict(previous_metrics(),buzz_current_public_capture_v39=dict(manifest=self.finished,
                actor_inputs_or_commands_changed=False,extra_camera_calls=0,extra_physics_ticks=0,
                native_integration='explicit fixed TRAIN diagnostic only; candidate helper retains no production admission',
                frame_sampling='first5 policy observations plus every10th; selected before outcomes',formal_metric=False))
        actor.metrics_report=MethodType(metrics,actor)
