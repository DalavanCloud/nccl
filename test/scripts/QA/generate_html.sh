#!/bin/bash

versions="$@"

base=$1

vers="$1"
shift
while [ "$1" != "" ]; do
vers="$vers / $1"
shift
done
 
ngpus=8
gpumodels=`ls $base/results`

mkdir -p comp
html=comp/index.html
cat > $html << EOF
<head><title>NCCL comparison - $vers</title></head>
<html>
<body>
<h1>NCCL comparison - $vers</h1>
EOF

for gpu in $gpumodels; do
  ./generate_plots.sh $gpu $ngpus $versions
  ./generate_diffs.sh $gpu $ngpus $versions
done

for op in all_reduce all_gather reduce_scatter reduce broadcast; do
  echo "<h2>$op</h2>" >> $html
  for gpu in $gpumodels; do
     echo "<h3>$gpu</h3>" >> $html
     echo "<table border=0><tr>" >> $html
     for ngpu in `seq 2 2 $ngpus`; do
        echo "<th>$ngpu GPUs</th>" >> $html
     done
     echo "</tr><tr>" >> $html
     for ngpu in `seq 2 2 $ngpus`; do
        echo "<td><a href=\"$gpu/$op-$ngpu.png\"><img width=300 src=\"$gpu/$op-$ngpu.png\"></a></td>" >> $html
     done
     echo "</tr><tr>" >> $html
     for ngpu in `seq 2 2 $ngpus`; do
        diff=`cat comp/$gpu/$op-$ngpu.diff`
        echo "<td>$diff</td>" >> $html
     done
     echo "</tr></table>" >> $html
  done
done

echo "</body></html>" >> $html
