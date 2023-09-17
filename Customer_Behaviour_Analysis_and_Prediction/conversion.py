# -*- coding: utf-8 -*-

####convert login data to usable data####

import sys

infile = sys.argv[1]
tcomp = -1
with open("company_log.csv","a") as g:
    with open(infile) as f:
        for line in f:
            fields = line.strip().split()
            comp = fields[1]
            if comp!= tcomp:
                tcomp = comp
                g.write("\n{},{}".format(tcomp,fields[4]))
            else:
                g.write(",{}".format(fields[4]))