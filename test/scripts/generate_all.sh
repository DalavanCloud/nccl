#!/bin/bash

DATE=$(cat $HOME/LASTLY_RUN | awk -F'.' '{ print $1 }')

VER=$1
if [ "$VER" == "" ]; then
   VER="master"
fi

version_checked=$(echo $VER | sed -e '/^[0-9]*\.[0-9]*\.[0-9]*$/d')
if [ -n "$version_checked" ]; then
  TAG=$DATE
else
  TAG=$VER
fi

DST=/var/www/html/nccl/nightly/$TAG
mkdir -p $DST

cd $HOME/install/nccl.deb/
rm -rf comp
GM=GROUP
PM=PARALLEL

for OPT in "" "_mpi" "_reorder" "_latency" "_mpi_latency"; do
   if [ "$OPT" == "_latency" ] || [ "$OPT" == "_mpi_latency" ]; then
     mode="lat"
   else
     mode="bw"
   fi
   ./generate_html.sh $mode $VER$GM$OPT $VER$PM$OPT 2.0.5$OPT 2.0.2$OPT 1.6.5$OPT 1.5.4$OPT
   DIR=html$OPT
   echo $DIR
   rm -rf $DST/$DIR
   mv comp $DST/$DIR
   chmod -R 775 $DST/$DIR
   # check failure
   grep -r -i "FAIL" ${VER}$GM$OPT/results >> $DST/$DIR/fail.txt
   grep -r -i "FAIL" ${VER}$PM$OPT/results >> $DST/$DIR/fail.txt
done

DIR=html_dataop
rm -rf $DST/$DIR
./dop_generate_html.sh ${VER}${GM}_all
mv data.op $DST/$DIR
grep -r -i "FAIL" ${VER}${GM}_all/results >> $DST/$DIR/fail.txt
grep -r -i "FAIL" ${VER}${PM}_all/results >> $DST/$DIR/fail.txt

for OPT in "_multinode" "_multinode_latency"; do
   if [ "$OPT" == "_multinode_latency" ]; then
     mode="lat"
   else
     mode="bw"
   fi
   ./multinode_generate_html.sh $mode $VER$GM$OPT $VER$PM$OPT 2.0.5$OPT 2.0.2$OPT
   DIR=html$OPT
   echo $DIR
   rm -rf $DST/$DIR
   mv multinode $DST/$DIR
   chmod -R 775 $DST/$DIR
   # check failure
   grep -r -i "FAIL" ${VER}$GM$OPT/results >> $DST/$DIR/fail.txt
   grep -r -i "FAIL" ${VER}$PM$OPT/results >> $DST/$DIR/fail.txt
done
