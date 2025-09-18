import os
import time
from concurrent.futures import ThreadPoolExecutor
from ..logging import logger

class TraceUploader:
    def __init__(self, bucket_name: str, run_id: str = None):
        self.bucket_name = bucket_name
        self.run_id = run_id or f"run_{int(time.time())}"
        self.upload_path = f"traces/{self.run_id}"
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def queue_upload(self, trace_path: str, memory_path: str, step: int, rank: int):
        """Queue files for background upload"""
        self.executor.submit(self._upload_file, trace_path, f"trace_rank_{rank}_step_{step}.json", step)
        self.executor.submit(self._upload_file, memory_path, f"memory_snapshot_rank{rank}_step{step}.pkl", step)
    
    def _upload_file(self, local_path: str, remote_name: str, step: int):
        """Upload a single file"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(f"{self.upload_path}/step_{step}/{remote_name}")
            
            blob.upload_from_filename(local_path)
            logger.info(f"✅ Uploaded {remote_name}")
            os.remove(local_path)
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {remote_name}: {e}")