
import sys
import math
import numpy as np

fp = open(sys.argv[1], "r");

p = [];
y = [];

for line in fp:
    line = line.strip()[:-1].split("\t");
    y.append(float(line[0]));
    p.append(float(line[1]));

r2 = np.corrcoef(p, y)[0, 1];
r2 = r2 * r2;

press = 0.0;
for i in range(len(y)):
   press += ( p[i] - y[i] ) * ( p[i] - y[i] );

rmsep = math.sqrt(press / len(y));

print("q2 = ", r2);
print("RMSE (cv) = ", rmsep);

