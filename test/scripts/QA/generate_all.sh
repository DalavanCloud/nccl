#!/bin/bash

VER=$1
if [ "$VER" == "" ]; then
   VER=2.0.4
fi

TAG=$VER
DST=/var/www/html/nccl/QA/$TAG
mkdir -p $DST

for OPT in "" "_mpi" "_reorder" ; do
   ./generate_html.sh $VER$OPT 2.0.2$OPT 1.6.5$OPT 1.5.4$OPT
   DIR=html$OPT
   echo $DIR
   rm -rf $DST/$DIR
   mv comp $DST/$DIR
   chmod -R 775 $DST/$DIR
   # check failure
   grep -r -i "FAIL" $VER$OPT/results >> $DST/$DIR/fail.txt
done

DIR=html_dataop
rm -rf $DST/$DIR
./dop_generate_html.sh ${VER}_all
mv data.op $DST/$DIR
grep -r -i "FAIL" ${VER}_all/results >> $DST/$DIR/fail.txt

DIR=html_multinode
rm -rf $DST/$DIR
./multinode_generate_html.sh ${VER}_multinode 2.0.2_multinode
mv multinode $DST/$DIR
grep -r -i "FAIL" ${VER}_multinode/results >> $DST/$DIR/fail.txt
