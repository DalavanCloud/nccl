#!/bin/bash

generate_perf() {
gpumodel=$1
op=$2
ngpus=$3
busbwcol=$4

result=results/$gpumodel/$op.$ngpus

mkdir -p results
echo "Running test/perf/${op}_perf -g $ngpus ..."
test/perf/${op}_perf -g $ngpus -b 5000 -e 955000 -i 5000 > $result.out
test/perf/${op}_perf -g $ngpus -b 1000000 -e 19000000 -i 1000000 >> $result.out
test/perf/${op}_perf -g $ngpus -b 20000000 -e 400000000 -i 20000000 >> $result.out

cat $result.out | grep float | awk "{ print \$1,\$$busbwcol; }" > $result.values

cat > $result.plot << EOF
set term png
set output "$result.png"
plot "$result.values" using 2:1, \
     "ref/1.6.1/$gpumodel/$op.$ngpus.values" using 2:1, \
     "ref/2.0.2/$gpumodel/$op.$ngpus.values" using 2:1
replot
EOF

gnuplot $result.plot
}

ngpu_loop() {
gpumodel=$1
op=$2
maxgpu=$3
busbwcol=$4
for ngpus in `seq 2 2 $maxgpu`; do
  generate_perf $gpumodel $op $ngpus $busbwcol
done
}

gpumodel=$1
maxgpu=$2

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <maxgpus>"
  exit 1
fi

ngpu_loop reduce $gpumodel $maxgpu 12
ngpu_loop broadcast $gpumodel $maxgpu 7
ngpu_loop all_reduce $gpumodel $maxgpu 11
ngpu_loop all_gather $gpumodel $maxgpu 10
ngpu_loop reduce_scatter $gpumodel $maxgpu 11
