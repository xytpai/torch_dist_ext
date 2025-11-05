import os
import torch


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)

CommWorkspace = eval(f"torch.classes.{package_name}.CommWorkspace")
allreduce_rms_fusion = eval(f"torch.ops.{package_name}.allreduce_rms_fusion")
