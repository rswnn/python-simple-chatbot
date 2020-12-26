import os
import subprocess

subprocess.call("rm model*", shell="True")
subprocess.call("rm data.pickle", shell="True")
subprocess.call("rm checkpoint", shell="True")

print("Directory cleaned")
