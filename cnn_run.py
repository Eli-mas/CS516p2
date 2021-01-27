import subprocess, sys

attr = sys.argv[1]

with open(f'out/{attr}.out','w') as f:
	print(subprocess.Popen(['python','cnn.py',attr], stdout=f, stderr=f).pid)

