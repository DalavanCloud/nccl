#!/bin/bash

mode=$1
shift 1

versions="$@"

base="$1"

vers="$1"
shift
while [ "$1" != "" ]; do
vers="$vers / $1"
shift
done
 
nnode=2
maxproc=16
maxthread=8
maxgpus=8
gpumodels=`ls $base/results`

mkdir -p multinode
html=multinode/index.html
cat > $html << EOF
<head><title>NCCL comparison - $vers</title></head>
<html>
<body>
<h1>NCCL comparison - $vers</h1>
EOF

for gpu in $gpumodels; do
  ./multinode_generate_plots.sh $mode $gpu $nnode $maxproc $maxthread $maxgpus $versions
#  ./generate_diffs.sh $gpu $nnode $maxproc $maxthread $maxgpus $versions
done

for op in all_reduce all_gather reduce_scatter reduce broadcast; do
  echo "<h2>$op</h2>" >> $html
  for gpu in $gpumodels; do
     echo "<h3>$gpu</h3>" >> $html
     echo "<table border=0><tr>" >> $html
     declare -i nproc=2
     while [[ $nproc -le $maxproc ]] ; do
       declare -i nthread=1
       while [[ $nthread -le $maxthread ]] ; do
         echo "</tr><tr>" >> $html
         declare -i ngpus=1
         while [[ $ngpus -le $maxgpus ]]; do
           ngperproc=$(expr $nthread \* $ngpus)
           if (( $( expr $nproc \* $ngperproc ) <= $maxproc )); then
              echo "<td> P$nproc x T$nthread x G$ngpus </td>" >> $html
           fi
           ngpus+=$ngpus
         done
         echo "</tr><tr>" >> $html
         declare -i ngpus=1
         while [[ $ngpus -le $maxgpus ]]; do
           ngperproc=$(expr $nthread \* $ngpus)
           if (( $( expr $nproc \* $ngperproc ) <= $maxproc )); then
              echo "<td><a href=\"$gpu/$op-$nnode-$nproc-$nthread-$ngpus.png\"><img width=300 src=\"$gpu/$op-$nnode-$nproc-$nthread-$ngpus.png\"></a></td>" >> $html
           fi
           ngpus+=$ngpus
         done
         echo "</tr><tr>" >> $html
         nthread+=$nthread
       done
       nproc+=$nproc
     done
     echo "</tr></table>" >> $html
  done
done

echo "</body></html>" >> $html
