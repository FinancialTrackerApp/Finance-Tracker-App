import torch
print(torch.__version__)        # e.g., 2.4.0+cu121
print(torch.version.cuda)       # should say 12.1
print(torch.cuda.is_available())# should be True
import sys
print(sys.executable)  # shows which Python your notebook is using

