from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
# from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor


jobstores = {
    'mongo': MongoDBJobStore(),
    # 'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
}
executors = {
    'default': ThreadPoolExecutor(20),
    'processpool': ProcessPoolExecutor(5)
}
job_defaults = {
    'coalesce': False,
    'max_instances': 3
}

def print1():
    print("Hello, this is a background task running every 10 seconds.")



scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc)


scheduler.add_job(print1, 'interval', seconds=2, id='print1_job', replace_existing=True)

def start_scheduler():
    """
    Start the background scheduler.
    """
    if not scheduler.running:
        print("Starting the scheduler...")
    scheduler.start()

def shutdown_scheduler():
    """
    Shutdown the background scheduler.
    """
    if scheduler.running:
        print("Shutting down the scheduler...")
        scheduler.shutdown(wait=False)
    else:
        print("Scheduler is not running.")