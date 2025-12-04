import sys
import os

output_path = sys.argv[1]

with open("torgo.csv", "r") as f:
    content = f.readlines()
f.close()

for lines in content:
    path, text, _ = lines.split(",")
    #print(path, text)

    filepath = path.replace(".wav",".txt")
    #print(filepath)

    with open(output_path+"/"+filepath, "w") as f:
        f.write(text)
    f.close()
