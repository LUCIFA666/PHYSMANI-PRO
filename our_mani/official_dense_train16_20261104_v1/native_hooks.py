"""Scoped same-instance forwarding; original methods execute exactly once.

No class/module monkeypatch, global simExtStep replacement, extra task.step or
clock repair. The observed methods are restored even after a failed attempt.
"""


class ScopedNativeStepObserver:
    def __init__(self,pyrep,task,ledger):
        self.pyrep,self.task,self.ledger=pyrep,task,ledger
        self.original_physics=pyrep.step;self.original_task=task.step;self.active=False
        if not callable(self.original_physics)or not callable(self.original_task):
            raise TypeError('Actual bound step functions required')

    def __enter__(self):
        if self.active:raise ValueError('Observer cannot be nested/reused')
        self.saved=[]
        def physics(*args,**kwargs):
            before=self.ledger.before_physics()
            value=self.original_physics(*args,**kwargs)
            self.ledger.after_physics(before)
            return value
        def task_step(*args,**kwargs):
            before=self.ledger.before_task_driver()
            value=self.original_task(*args,**kwargs)
            self.ledger.after_task_driver(before)
            return value
        self.wrappers=(physics,task_step)
        for obj,fn in ((self.pyrep,physics),(self.task,task_step)):
            self.saved.append((obj,'step' in vars(obj),vars(obj).get('step')));setattr(obj,'step',fn)
        self.active=True
        return self

    def __exit__(self,kind,error,traceback):
        try:
            if self.ledger.fault is None:
                try:self.ledger.flush('attempt_return'if kind is None else'exception_boundary_partial')
                except Exception:
                    if kind is None:raise
        finally:
            for obj,had,value in reversed(self.saved):
                if had:setattr(obj,'step',value)
                else:delattr(obj,'step')
            self.active=False
        return False
