#!/bin/bash

generate_plot() {
gpumodel=$1
ngpus=$2
op=$3
busbwcol=$4
shift 4

sum() {
total=0
while read line; do
  set -- $line
  if [ "$2" != "" ]; then
    total=`echo " $total + $2 " | bc -l`
  fi
done
echo $total
}

graph=comp/$gpumodel/$op-$ngpus

firstversion=1
for version in $@; do
  data=$version/results/$gpumodel/$op.$ngpus
  cat $data.out | grep float | awk "{ print \$1,\$$busbwcol; }" > $data.values
  score=`awk '{ sum += $2 } END { print sum }' $data.values`
  if [ "$firstversion" == "1" ]; then
    firstversion=0
    refscore=$score
    echo "$version : $score" > $graph.diff
  else
    echo -n "<br>$version : $score, " >> $graph.diff
    diff=`echo " ($score / $refscore - 1) * 100 " | bc -l | xargs printf "%.2f"`
    if [ "`echo $diff | cut -c 1`" == "-" ]; then
      if [ "`echo $diff | cut -c 2- | cut -d '.' -f 1`" -gt "20" ]; then
        echo "<b><font color=\"#00cc00\">$diff %</font></b>" >> $graph.diff
      else
        echo "<font color=\"#006600\">$diff %</font>" >> $graph.diff
      fi
    else
      if [ "`echo $diff | cut -d '.' -f 1`" -gt "20" ]; then
        echo "<b><font color=\"#ff0000\">$diff %</font></b>" >> $graph.diff
      else
        echo "<font color=\"#660000\">$diff %</font>" >> $graph.diff
      fi
    fi
  fi
  echo -n "."
done
}

plot_ngpu_loop() {
gpumodel=$1
maxgpu=$2
shift 2
for ngpus in `seq 2 2 $maxgpu`; do
  generate_plot $gpumodel $ngpus $@
done
}

gpumodel=$1
maxgpu=$2
if [ "$maxgpu" == "" ]; then
  echo "Usage $0 <gpumodel> <ngpus>"
  exit 1
fi

shift 2
mkdir -p comp/$gpumodel
echo -n "Generating diffs "
plot_ngpu_loop $gpumodel $maxgpu reduce 12 $@
plot_ngpu_loop $gpumodel $maxgpu broadcast 7 $@
plot_ngpu_loop $gpumodel $maxgpu all_reduce 11 $@
plot_ngpu_loop $gpumodel $maxgpu all_gather 10 $@
plot_ngpu_loop $gpumodel $maxgpu reduce_scatter 11 $@

echo "done."
