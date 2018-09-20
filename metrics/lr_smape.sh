#!/usr/bin/env bash
dir=$1
tool=$2
metricdir=`dirname "$0"`
if [[ $tool == "stan" ]]; then
    weight=`grep -ir  "se_mean" -A4 $dir/pplout* | tail -4 | awk '{print $2}' | head -1`
    bias=`grep -ir  "se_mean" -A4 $dir/pplout* | tail -4 | awk '{print $2}' | head -2 | tail -1`
    res=`$metricdir/lr_smape.py 0.1 $weight $bias $dir`
elif [[ $tool == "edward" ]]; then
    weight=`cat $dir/edwardout_* | tail -3 | head -1 | grep -io "\-\?[0-9]\+[\.]\?[0-9]\+"`
    bias=`cat $dir/edwardout_* | tail -2 | head -1 | grep -io "\-\?[0-9]\+[\.]\?[0-9]\+"`
    res=`$metricdir/lr_smape.py 0.1 $weight $bias $dir`
elif [[ $tool == "pyro" ]];then
    weight=`cat $dir/pyroout_* | grep 'w_mean' | awk '{print $3}' | grep -io "\-\?[0-9\.]\+"`
    bias=`cat $dir/pyroout_* |grep 'b_mean' | awk '{print $3}' | grep -io "[0-9\.]\+"`
    res=`$metricdir/lr_smape.py 0.1 $weight $bias $dir`
fi

echo $res

	 
    

