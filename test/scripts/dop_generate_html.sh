#!/bin/bash

version="$1"

ngpus=8
gpumodels=`ls $version/results`

mkdir -p data.op
html=data.op/index.html
cat > $html << EOF
<head><title>NCCL comparison - $version</title></head>
<html>
<body>
<h1>NCCL comparison - $version</h1>
EOF

for gpu in $gpumodels; do
  ./dop_generate_plots.sh $gpu $ngpus $version
done

for op in all_reduce all_gather reduce_scatter reduce broadcast; do
  echo "<h2>$op</h2>" >> $html
  for gpu in $gpumodels; do
     echo "<h3>$gpu</h3>" >> $html
     echo "<table border=0><tr>" >> $html
     echo "<th>      </th>" >> $html
     for ngpu in `seq 2 2 $ngpus`; do
        echo "<th>$ngpu GPUs</th>" >> $html
     done
     echo "</tr><tr>" >> $html
     for pow in pow2 npow2; do
       echo "<td> $pow </td>" >> $html
       for ngpu in `seq 2 2 $ngpus`; do
          echo "<td><a href=\"$gpu/$pow/$op-$ngpu.png\"><img width=300 src=\"$gpu/$pow/$op-$ngpu.png\"></a></td>" >> $html
       done
       echo "</tr><tr>" >> $html
     done
     echo "</tr></table>" >> $html
  done
done

echo "</body></html>" >> $html
