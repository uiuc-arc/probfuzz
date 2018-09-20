#!/usr/bin/env bash
dir=$1
tool=$2
metricdir=`dirname "$0"`
weight=`grep -ir  "se_mean" -A4 $dir/pplout* | tail -4 | awk '{print $2}' | head -1`
bias=`grep -ir  "se_mean" -A4 $dir/pplout* | tail -4 | awk '{print $2}' | head -2 | tail -1`
edward_weight=`cat $dir/edwardout_* | tail -2 | head -1 | grep -io "\-\?[0-9]\+[\.]\?[0-9]\+"`
edward_bias=`cat $dir/edwardout_* | tail -1 | head -1 | grep -io "\-\?[0-9]\+[\.]\?[0-9]\+"`
pyro_weight=`cat $dir/pyroout_* | grep 'w_mean' -A1 | awk '{print $3}' | grep -io "\-\?[0-9\.]\+"`
pyro_bias=`cat $dir/pyroout_* |grep 'b_mean' -A1 | awk '{print $3}' | grep -io "\-\?[0-9\.]\+"`

if [[ $tool == "stan" ]]; then
    res1=`$metricdir/simple_smape.py 0.1 $weight $bias $edward_weight $edward_bias $dir`
    res2=`$metricdir/simple_smape.py 0.1 $weight $bias $pyro_weight $pyro_bias $dir`
elif [[ $tool == "edward" ]]; then
    res1=`$metricdir/simple_smape.py 0.1 $edward_weight $edward_bias $weight $bias $dir`
    res2=`$metricdir/simple_smape.py 0.1 $edward_weight $edward_bias $pyro_weight $pyro_bias $dir`
elif [[ $tool == "pyro" ]];then
    res1=`$metricdir/simple_smape.py 0.1 $pyro_weight $pyro_bias $edward_weight $edward_bias $dir`
    res2=`$metricdir/simple_smape.py 0.1 $pryo_weight $pyro_bias $weight $bias $dir`
fi

if [[ $res1 == *"False"* && $res2 == *"False"* ]]; then
    echo "False"
else
    echo "True"
fi

