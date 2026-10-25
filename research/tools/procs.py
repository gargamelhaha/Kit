import os, sys
me = {os.getpid(), os.getppid()}
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) in me: continue
    try: a = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
    except Exception: continue
    if any(k in a for k in sys.argv[1:]): print(pid, a[:110])
