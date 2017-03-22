#!/usr/bin/python

import os,sys

h2=1
h3=1
h4=1

for line in open(sys.argv[1]).readlines() :
  if "<h2>" in line :
    label = "%d" % h2;
    line = line.replace("<h2>", "<a name=\"#%s\"></a><h2>%s " % (label, label))
    h2+=1
    h3=1
    h4=1
  if "<h3>" in line :
    label = "%d.%d" % (h2,h3)
    line = line.replace("<h3>", "<a name=\"#%s\"></a><h3>%s " % (label,label))
    h3+=1
    h4=1
  if "<h4>" in line :
    label = "%d.%d.%d" % (h2,h3,h4)
    line = line.replace("<h4>", "<a name=\"#%s\"></a><h4>%s " % (label,label))
    h4+=1
  print line,
