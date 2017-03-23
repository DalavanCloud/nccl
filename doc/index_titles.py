#!/usr/bin/python

import os,sys

h2=0
h3=0
h4=0
contents=""

for line in open(sys.argv[1]).readlines() :
  if "<h1>" in line :
    label = "top"
    title = (line.split("<h1>")[1]).split("</h1>")[0]
    line = line.replace("<h1>", "<a name=\"%s\"></a><h1>" % label)
    contents += "<p><b><a href=\"#%s\">%s</a></b><p>\n" % (label, title)
  if "<h2>" in line :
    h2+=1
    h3=1
    h4=1
    label = "%d" % h2;
    title = (line.split("<h2>")[1]).split("</h2>")[0]
    line = line.replace("<h2>", "<a name=\"%s\"></a><h2>%s " % (label, label))
    contents += "<hr><a href=\"#%s\">%s %s</a><br>\n" % (label, label, title)
  if "<h3>" in line :
    h3+=1
    h4=1
    label = "%d.%d" % (h2,h3)
    title = (line.split("<h3>")[1]).split("</h3>")[0]
    line = line.replace("<h3>", "<a name=\"%s\"></a><h3>%s " % (label,label))
    contents += "&nbsp;&nbsp;<a href=\"#%s\">%s %s</a><br>\n" % (label, label, title)
  if "<h4>" in line :
    h4+=1
    label = "%d.%d.%d" % (h2,h3,h4)
    title = (line.split("<h4>")[1]).split("</h4>")[0]
    line = line.replace("<h4>", "<a name=\"%s\"></a><h4>%s " % (label,label))
    contents += "&nbsp;&nbsp;&nbsp;&nbsp;<a href=\"#%s\">%s %s</a><br>\n" % (label, label, title)
  if "$CONTENTS" in line :
    line = line.replace("$CONTENTS", contents)
  print line,
