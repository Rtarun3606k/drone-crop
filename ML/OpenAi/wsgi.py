from app import app
from CornJob.JobSchudler import start_scheduler, shutdown_scheduler
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Register signal handlers
       
        
        # Start the background scheduler
        start_scheduler()
        
        # Start the Flask app
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
        shutdown_scheduler()
    except Exception as e:
        logger.error(f"Application error: {e}")
        shutdown_scheduler()
        raise
    finally:
        shutdown_scheduler()

