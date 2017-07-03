#!/bin/bash

generate_plot() {
op=$1
busbwcol=$2
gpumodel=$3
nnode=$4
nproc=$5
nthread=$6
ngpus=$7
shift 7

graph=multinode/$gpumodel/$op-$nnode-$nproc-$nthread-$ngpus
cat > $graph.plot << EOF
set term png
set terminal png size 1280,1024
set output "$graph.png"
set title "$op, P$nproc x T$nthread x G$ngpus, $gpumodel" noenhanced
set logscale x
EOF

firstline=1
for version in $@; do
  data=$version/results/$gpumodel/$op.$nproc.$nthread.$ngpus
  cat $data.out | grep float | awk "{ print \$1,\$$busbwcol; }" > $data.values
  if [ "$firstline" == "1" ]; then
    firstline=0
    echo -n "plot " >> $graph.plot
  else
    echo ", \\" >> $graph.plot
  fi
  echo -n "\"$data.values\" using 1:2 with lines title \"$version\"" >> $graph.plot
  echo -n "."
done
echo "" >> $graph.plot

echo "replot" >> $graph.plot
gnuplot $graph.plot
}

plot_ngpu_loop() {
op=$1
col=$2
gpumodel=$3
nnode=$4
maxproc=$5
maxthread=$6
maxgpu=$7
shift 7

declare -i nproc=2
while [[ $nproc -le $maxproc ]] ; do
  declare -i nthread=1
  while [[ $nthread -le $maxthread ]] ; do
    declare -i ngpus=1
    while [[ $ngpus -le $maxgpu ]]; do
      ngperproc=$(expr $nthread \* $ngpus)
      if (( $( expr $nproc \* $ngperproc ) <= $maxproc )); then
        echo "Running test/perf/${op}_perf on $nnode nodes, $nproc processes, each process having $nthread threads with $ngpus GPUs ..."
        generate_plot $op $col $gpumodel $nnode $nproc $nthread $ngpus $@
      fi
      ngpus+=$ngpus
    done
    nthread+=$nthread
  done
  nproc+=$nproc
done

}

gpumodel=$1

shift 1
mkdir -p multinode/$gpumodel
echo -n "Generating images "
plot_ngpu_loop reduce 12 $gpumodel $@
plot_ngpu_loop broadcast 7 $gpumodel $@
plot_ngpu_loop all_reduce 11 $gpumodel $@
plot_ngpu_loop all_gather 10 $gpumodel $@
plot_ngpu_loop reduce_scatter 11 $gpumodel $@

echo "done."
