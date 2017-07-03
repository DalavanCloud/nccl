#!/bin/bash

generate_plot() {
gpumodel=$1
ngpus=$2
op=$3
busbwcol=$4
pow=$5
version=$6

graph=data.op/$gpumodel/$pow/$op-$ngpus
cat > $graph.plot << EOF
set term png
set terminal png size 1280,1024
set output "$graph.png"
set title "$op, $ngpus x $gpumodel" noenhanced
set logscale x
set key left top
EOF

firstline=1
path=${version}/results/$gpumodel/$pow
for file in $path/$op.$ngpus.*.out; do
  cat $file | grep 'e-\|e+' | awk "{ print \$1,\$$busbwcol; }" > $file.values
  if [ "$firstline" == "1" ]; then
    firstline=0
    echo -n "plot " >> $graph.plot
  else
    echo ", \\" >> $graph.plot
  fi
  fn="${file##*/}"
  legend=$( echo $fn | cut -d '.' -f 3,4 )
  echo -n "\"$file.values\" using 1:2 with lines title \"$legend\"" >> $graph.plot
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

gpumodel=$1
maxgpu=$2
if [ "$maxgpu" == "" ]; then
  echo "Usage $0 <gpumodel> <ngpus>"
  exit 1
fi

shift 2
for pow in pow2 npow2; do
  mkdir -p data.op/$gpumodel/$pow
  echo -n "Generating images "
  plot_ngpu_loop $gpumodel $maxgpu reduce 12 $pow $@
  plot_ngpu_loop $gpumodel $maxgpu broadcast 7 $pow $@
  plot_ngpu_loop $gpumodel $maxgpu all_reduce 11 $pow $@
  plot_ngpu_loop $gpumodel $maxgpu all_gather 10 $pow $@
  plot_ngpu_loop $gpumodel $maxgpu reduce_scatter 11 $pow $@
done

echo "done."
