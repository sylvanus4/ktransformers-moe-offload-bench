import os, sys, importlib
variant = sys.argv[1] if len(sys.argv) > 1 else "amx"
os.environ["KT_KERNEL_CPU_VARIANT"] = variant
os.environ["KT_KERNEL_DEBUG"] = "1"
import kt_kernel
ext = kt_kernel.kt_kernel_ext
print("variant requested:", variant)
print("ext module:", ext.__name__)
moe = getattr(ext, "moe", None)
if moe:
    cls = [a for a in dir(moe) if "MOE" in a.upper() or "MoE" in a]
    print("MOE classes:", cls)
print("ext top:", [a for a in dir(ext) if not a.startswith("_")][:15])
