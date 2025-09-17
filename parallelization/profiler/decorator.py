import os
import time
import torch
from functools import wraps
from torch.profiler import profile, ProfilerActivity, schedule
import torch.cuda.memory as _cm
from ..logging import logger
import wandb
import torch.distributed as dist
from torch.distributed.tensor.debug import CommDebugMode


class performance_monitor:
    """
    A unified decorator for PyTorch performance monitoring, combining the PyTorch
    profiler, a FLOP counter, and a communication logger.

    This decorator is stateful and designed to wrap a single training step function.
    It manages the profiling lifecycle automatically.

    Usage:
        monitor = performance_monitor(
            model,
            comm_mode=comm_mode_obj,
            flop_counter_step=5,
            comm_logger_step=2
        )

        @monitor
        def train_step(batch):
            # ... training logic ...

        for batch in dataloader:
            train_step(batch)
    """
    def __init__(
        self,
        model,
        comm_mode=None,
        enabled=True,
        output_dir="./profiler_logs",
        skip_first=1,
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
        with_stack=False,
        with_flops=True,
        with_modules=False,
        nsight_enabled=False,
        nsight_output="nsight_profile",
        flop_counter_step=None,
        comm_logger_step=None,
    ):
        self.model = model
        self.comm_mode = comm_mode
        self.enabled = enabled
        self.output_dir = output_dir
        self.schedule_args = {
            "skip_first": skip_first, "wait": wait, "warmup": warmup,
            "active": active, "repeat": repeat
        }
        self.profiler_args = {
            "with_stack": with_stack, "with_flops": with_flops, "with_modules": with_modules
        }
        self.nsight_enabled = nsight_enabled
        self.nsight_output = nsight_output
        self.flop_counter_step = flop_counter_step
        self.comm_logger_step = comm_logger_step
        self.comm_mode = CommDebugMode() if self.comm_logger_step is not None else None

        self.prof = None
        self.step_num = 0
        self.rank = int(os.environ.get("LOCAL_RANK", 0))

    def _setup_profiler(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.nsight_enabled:
            try:
                torch.cuda.cudart().cudaProfilerStart()
                logger.info(f"🔍 NSight profiling enabled - output: {self.nsight_output}")
            except Exception as e:
                logger.warning(f"⚠️ NSight profiling failed to start: {e}")

        _cm._record_memory_history()

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        profiler_schedule = schedule(**self.schedule_args)

        logger.info(f"📊 PyTorch profiler enabled: {self.schedule_args}")

        self.prof = profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            **self.profiler_args
        )
        self.prof.__enter__()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            if self.prof is None:
                self._setup_profiler()

            # Define how to run the user's training step, applying the comm_mode context if enabled
            def execute_step():
                if self.comm_mode:
                    with self.comm_mode:
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            # Flop counter logic (runs on a specific step)
            if self.step_num == self.flop_counter_step:
                result = self._run_flop_counter(execute_step)
            else:
                # Regular execution for other steps
                result = execute_step()

            # Communication logger logic (runs after the step)
            if self.step_num == self.comm_logger_step:
                self._run_comm_logger()
            
            self.prof.step()
            self.step_num += 1
            return result
        return wrapper

    def _run_flop_counter(self, execute_func):
        from torch.utils.flop_counter import FlopCounterMode
        logger.info(f"FLOP counter running for step {self.step_num}...")
        with FlopCounterMode(mods=self.model, display=True, depth=None) as ftdm:
            t = time.time()
            result = execute_func()
            t_lapsed = time.time() - t
            
            # Sum flops from all modules to be robust to model name changes
            total_flops = sum(sum(m.values()) for m in ftdm.flop_counts.values())
            tflops = total_flops / t_lapsed / 1e12 if t_lapsed > 0 else 0
            logger.info(f"rank {self.rank} step {self.step_num} total_flops: {total_flops:,} tflops: {tflops:.2f}")
            if wandb.run is not None:
                # flop counter
                wandb.log({
                    "prof/step": self.step_num,
                    f"prof/tflops_rank_{self.rank}": tflops,
                    f"prof/total_flops_rank_{self.rank}": total_flops,
                }, commit=False)   # or step=global_step if you prefer to align, but keep commit=False
        return result

    def _run_comm_logger(self):
        if self.comm_mode and self.rank == 0 and dist.is_initialized():
            logger.info(f"📝 Logging communication debug info for step {self.step_num}...")
            try:
                self.comm_mode.log_comm_debug_tracing_table_to_file(
                    noise_level=1,
                    file_name=f"comm_debug_{self.step_num}.txt"
                )
                self.comm_mode.generate_json_dump(noise_level=2)
                logger.info("✅ Communication debug info saved.")
            except Exception as e:
                logger.error(f"⚠️ Failed to log communication debug info: {e}")

    def __del__(self):
        if self.prof:
            self.prof.__exit__(None, None, None)
        if self.nsight_enabled:
            try:
                torch.cuda.cudart().cudaProfilerStop()
                logger.info(f"✅ NSight profiling completed")
            except Exception as e:
                logger.warning(f"⚠️ NSight profiling failed to stop: {e}")


def trace_handler(prof):
    rank = int(os.environ.get("RANK", 0))  # use global rank
    if rank == 0:
        logger.info("\n📈 Profiling Summary:")
        output = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        logger.info(output)

        if wandb.run is not None:
            profile_metrics = {"prof/step": int(prof.step_num)}
            for event in prof.key_averages(group_by_input_shape=True)[:10]:
                metric = event.key.replace(" ", "_").replace("=", "").replace(",", "")
                cuda_ms = getattr(event, "cuda_time_total", 0) / 1000
                cpu_ms  = getattr(event, "cpu_time_total", 0) / 1000
                if cuda_ms > 0:
                    profile_metrics[f"prof/{metric}/cuda_time_total_ms"] = cuda_ms
                if cpu_ms > 0:
                    profile_metrics[f"prof/{metric}/cpu_time_total_ms"] = cpu_ms
            wandb.log(profile_metrics, commit=False)
        else:
            logger.warning("Wandb is not initialized")
    
    # Each rank saves its own trace
    trace_path = f"/tmp/trace_rank_{rank}_step_{prof.step_num}.json"
    prof.export_chrome_trace(trace_path)
    logger.info(f"💾 Chrome trace saved to: {trace_path}")

    # ─── export memory timeline as a Perfetto‐readable counter track ───
    snap_path = f"/tmp/memory_snapshot_rank{rank}_step{prof.step_num}.pkl"
    _cm._dump_snapshot(snap_path)
    logger.info(f"💾 Memory timeline (Allocated vs Reserved) saved to: {snap_path}")

    # # Compute total FLOPs
    total_flops = sum(evt.flops for evt in prof.key_averages() if hasattr(evt, "flops"))
    logger.info(f"\n💯 Total FLOPs (counted ops): {rank} {total_flops:,}") 