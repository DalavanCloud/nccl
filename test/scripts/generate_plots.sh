#!/bin/bash

generate_plot() {
gpumodel=$1
ngpus=$2
op=$3
busbwcol=$4
shift 4

graph=comp/$gpumodel/$op-$ngpus
cat > $graph.plot << EOF
set term png
set terminal png size 1280,1024
set output "$graph.png"
set title "$op, $ngpus x $gpumodel" noenhanced
set logscale x
set yrange [0:]
EOF

firstline=1
for version in $@; do
  data=$version/results/$gpumodel/$op.$ngpus
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
gpumodel=$1
maxgpu=$2
shift 2
for ngpus in `seq 2 2 $maxgpu`; do
  generate_plot $gpumodel $ngpus $@
done
}

mode=$1
gpumodel=$2
maxgpu=$3
if [ "$maxgpu" == "" ]; then
  echo "Usage $0 <mode> <gpumodel> <ngpus>"
  exit 1
fi

shift 3
mkdir -p comp/$gpumodel
echo -n "Generating images "
if [ "$mode" == "bw" ]; then
  plot_ngpu_loop $gpumodel $maxgpu reduce 12 $@
  plot_ngpu_loop $gpumodel $maxgpu broadcast 7 $@
  plot_ngpu_loop $gpumodel $maxgpu all_reduce 11 $@
  plot_ngpu_loop $gpumodel $maxgpu all_gather 10 $@
  plot_ngpu_loop $gpumodel $maxgpu reduce_scatter 11 $@
elif [ "$mode" == "lat" ]; then
  plot_ngpu_loop $gpumodel $maxgpu reduce 10 $@
  plot_ngpu_loop $gpumodel $maxgpu broadcast 5 $@
  plot_ngpu_loop $gpumodel $maxgpu all_reduce 9 $@
  plot_ngpu_loop $gpumodel $maxgpu all_gather 8 $@
  plot_ngpu_loop $gpumodel $maxgpu reduce_scatter 9 $@
else
  echo "Invalid mode. Please specify bw or lat"
fi
echo "done."
