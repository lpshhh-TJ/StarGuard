import subprocess, os
files = subprocess.check_output(['git', 'ls-files']).decode().splitlines()
total = 0
for f in files:
    try:
        with open(f, 'r', errors='ignore') as fh:
            total += sum(1 for _ in fh)
    except: pass
print(total)
