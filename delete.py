import glob
import os
for i in glob.glob("results/*"):
    bb = []
    for j in glob.glob(os.path.join(i, "sfiii3n", "sr6_128x4_das_nc", "model", "*.zip")):
        basename = os.path.basename(j)
        sample = int(basename.split(".")[0].split("_")[-1])
        bb.append((j, sample))
    if len(bb) == 0:
        continue
    maxSample = max(bb, key=lambda x:x[0])[-1]
    print(maxSample)
    keepTarget = sorted([j for  (j, sample) in bb if not (sample != maxSample and sample % 10000_000 != 0)])
    removeTarget = [j for (j, sample) in bb if j not in keepTarget]
    import shutil
    for i in removeTarget:
        os.remove(i)