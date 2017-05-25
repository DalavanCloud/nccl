#!/bin/bash

generate_perf() {
op=$1
ngpus=$2

if [ "$ngpus" == "" ]; then
  echo "Usage : $0 <operation> <ngpus>"
  exit 1
fi

result=$op.$ngpus

mkdir -p results
echo "Running test/perf/${op}_perf -g $ngpus ..."
test/perf/${op}_perf -g $ngpus -b 5000 -e 955000 -i 5000 > results/$result.out
test/perf/${op}_perf -g $ngpus -b 1000000 -e 19000000 -i 1000000 >> results/$result.out
test/perf/${op}_perf -g $ngpus -b 20000000 -e 400000000 -i 20000000 >> results/$result.out

cat results/$result.out | grep sum | awk '{ print $1,$12; }' > results/$result.values

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

for op in reduce broadcast all_reduce all_gather reduce_scatter; do
  for ngpus in 2 4 6 8; do
    generate_perf $op $ngpus
  done
done
