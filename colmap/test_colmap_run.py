import subprocess
r = subprocess.run(["colmap", "help"], capture_output=True, text=True)
print("stdout:", r.stdout[:300])
print("stderr:", r.stderr[:300])
print("rc:", r.returncode)
