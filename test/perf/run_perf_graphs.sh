#!/bin/bash

generate_perf() {
op=$1
ngpus=$2
busbwcol=$3

result=$op.$ngpus

mkdir -p results
echo "Running test/perf/${op}_perf -g $ngpus ..."
test/perf/${op}_perf -g $ngpus -b 5000 -e 955000 -i 5000 > results/$result.out
test/perf/${op}_perf -g $ngpus -b 1000000 -e 19000000 -i 1000000 >> results/$result.out
test/perf/${op}_perf -g $ngpus -b 20000000 -e 400000000 -i 20000000 >> results/$result.out

cat results/$result.out | grep float | awk "{ print \$1,\$$busbwcol; }" > results/$result.values

cat > results/$result.plot << EOF
set term png
set output "results/$result.png"
plot "results/$result.values" using 2:1, \
     "ref/1.6.1/$result.values" using 2:1, \
     "ref/2.0.2/$result.values" using 2:1
replot
EOF

gnuplot results/$result.plot
}

ngpu_loop() {
op=$1
maxgpu=$2
busbwcol=$3
for ngpus in `seq 2 2 $maxgpu`; do
  generate_perf $op $ngpus $busbwcol
done
}

maxgpu=$1

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <maxgpus>"
  exit 1
fi

ngpu_loop reduce $maxgpu 12
ngpu_loop broadcast $maxgpu 7
ngpu_loop all_reduce $maxgpu 11
ngpu_loop all_gather $maxgpu 10
ngpu_loop reduce_scatter $maxgpu 11
