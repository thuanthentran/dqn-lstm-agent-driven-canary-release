import os
import glob
import re

for file in glob.glob("gitops/releases/*-values.yaml"):
    with open(file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    port = None
    in_readiness = False
    
    for line in lines:
        if "readinessProbe:" in line:
            in_readiness = True
            new_lines.append(line)
        elif in_readiness and "exec:" in line:
            continue
        elif in_readiness and "grpc_health_probe" in line:
            m = re.search(r"-addr=:(\d+)", line)
            if m:
                port = m.group(1)
                new_lines.append(f"  grpc:\n    port: {port}\n")
            in_readiness = False
        else:
            new_lines.append(line)
            
    with open(file, "w") as f:
        f.writelines(new_lines)
