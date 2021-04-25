from .abstract_scheduler import AbstractScheduler

schedulers = {}

def register_scheduler(key, scheduler_class, *args, **kwargs):
    if not issubclass(scheduler_class, AbstractScheduler):
        print('scheduler must derived from AbstractScheduler')
        assert False
    if key not in schedulers:
        schedulers[key] = lambda: scheduler_class(*args, **kwargs)
        print('regestering', scheduler_class)

def unregister_scheduler(key):
    if key in schedulers:
        del schedulers[key]

def get_schedulers():
    return schedulers.values()
