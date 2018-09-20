#!/usr/bin/env bash
dir=$1
tool=$2
metricdir=`dirname "$0"`
if [[ $tool == "stan" ]]; then
    weight=`cat $dir/pplout* | grep "w\[" | awk '{ print $2 }' | tr '\n' ','`
    bias=`cat $dir/pplout* | grep "se_mean" -A100 |  grep "^b" | awk '{print $2}'`
    res=`$metricdir/mlr_smape.py 0.1 "$weight" "$bias" $dir`
elif [[ $tool == "edward" ]]; then
    weight=`grep -zo "\[\[.*\]\]" $dir/edwardout_* | tr '\n' ' '|  sed -e 's/\[\|\]\|(\|)//g' | sed -e 's/\s\+/,/g'`
    bias=`cat $dir/edwardout_* | tail -2 | head -1 | grep -io "[0-9]\+[\.]\?[0-9]\+"`
    res=`$metricdir/mlr_smape.py 0.1 "$weight" "$bias" $dir`
elif [[ $tool == "pyro" ]];then
    weight=`cat $dir/pyroout* | grep -zo  'w_mean.*\]\]))' | grep -zo '(\[.*\])' | tr '\n' ' '| sed -e 's/\[\|\]\|(\|)//g'`
    bias=`cat $dir/pyroout* | grep 'b_mean.*]))' | grep -o '(\[.*\])' | grep -o '[0-9]*\.\?[0-9]*'`
    res=`$metricdir/mlr_smape.py 0.1 "$weight" "$bias" $dir`
fi

echo $res

	 
    

