"""Run core/run_accuracy.py unchanged but with THREADS=3 instead of the hardcoded 12
(module constant plus the `threads=` defaults bound at def time). Usage: run_accuracy_wrap.py CORE_DIR [run_accuracy args]"""
import sys, os, inspect, runpy
core = sys.argv[1]; sys.path.insert(0, core)
import core_lib as K
N = int(os.environ.get("WRAP_THREADS", "3"))
K.THREADS = N
for name, f in list(vars(K).items()):
    if inspect.isfunction(f) and f.__module__ == "core_lib" and f.__defaults__:
        sig = inspect.signature(f); d = list(f.__defaults__)
        params = list(sig.parameters.values())[-len(d):]
        f.__defaults__ = tuple(N if (p.name in ("threads", "workers") and v == 12) else v for p, v in zip(params, d))
sys.argv = [os.path.join(core, "run_accuracy.py")] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
