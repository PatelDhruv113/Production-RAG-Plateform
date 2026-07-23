import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Observability tool to record execution latency for each phase of the RAG pipeline.
    """
    def __init__(self):
        self.metrics = {}

    def start_timer(self):
        return time.perf_counter()

    def stop_timer(self, start_time):
        return round((time.perf_counter() - start_time) * 1000, 2)  # Latency in ms

    @contextmanager
    def measure(self, phase_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            self.metrics[f"{phase_name}_ms"] = elapsed_ms
            logger.info(f"[MetricsLogger] Phase '{phase_name}' completed in {elapsed_ms} ms")

    def get_metrics(self):
        return self.metrics.copy()
