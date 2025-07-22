from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.executors.pool import ProcessPoolExecutor
from pytz import utc
import atexit
import signal
import sys
import logging
from CornJob.CornJobFunction import job_generate_speech_and_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Only one executor since ML is CPU-bound
executors = {
    'processpool': ProcessPoolExecutor(5)
}

# Job defaults: max_instances=1 prevents concurrent runs
job_defaults = {
    'coalesce': False,       # don’t “catch up” missed runs
    'max_instances': 1,      # only one running at a time
}

scheduler = BackgroundScheduler(
   jobstores={'default': MongoDBJobStore(host='localhost', port=27017, database='droneCrop', collection='translation_jobs')},
    executors=executors,
    job_defaults=job_defaults,
    timezone=utc
)

# Schedule your ML runner every 10 minutes
scheduler.add_job(
    job_generate_speech_and_report,
    trigger='interval',
    # minutes=10,
    seconds=5,
    id='Language_speeech_services',
    executor='processpool',
    misfire_grace_time=3600,  # if a run is delayed, allow up to 1 hour late
    replace_existing=True
)

def shutdown_scheduler():
    logger.info("Shutting down scheduler; waiting for running jobs to finish…")
    scheduler.shutdown(wait=True)  # <-- this waits for  ML job to complete

def signal_handler(signum, frame):
    shutdown_scheduler()
    sys.exit(0)

def start_scheduler():
    # register cleanup handlers
    atexit.register(shutdown_scheduler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    scheduler.start()
    logger.info("Scheduler started.")

if __name__ == "__main__":
    start_scheduler()
    # Keep main thread alive…
    signal.pause()