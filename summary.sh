#!/usr/bin/env bash
#usage ./summary.sh -m [metric] -d [directory of results]
metric="lr_smape"
directory="."
basedir=`pwd`
while getopts ":m:d:" opt;do
    case ${opt} in
	m )
	    metric=$OPTARG
	    ;;
	d )
	    directory=$OPTARG
	    ;;
	\? )
	    echo "Invalid usage"
	    echo "usage: ./summary.sh -m [metric] -d [directoryname]"
	    exit 1
	    ;;
	: )
	    echo "missing option"
	    echo "usage: ./summary.sh -m [metric] -d [directoryname]"
	    exit 1
	    ;;
    esac   
done

cd $directory
printf "Program,Stan_Crash,Stan_Num,Stan_Acc,Pyro_Crash,Pyro_Num,Pyro_Acc,Edward_Crash,Edward_Num,Edward_Acc\n"
for x in */; do
    printf "$x,"
    if [ -e $x/pplout* ]; then
	crash=`grep -ir "se_mean" $x/pplout* -L | wc -l`
	inf=`grep -ir "se_mean" $x/pplout* -A100 | grep -iw "inf\|nan" -l | wc -l`
	if [ $crash -eq 1 ];
	then	    
	    printf "*,"
	else
	    printf -- '-,'
	fi
	
	if [ $inf -ge 1 ];
	then	    
	    printf "*,"
	else
	    printf -- '-,'
	fi

	# acc check

	if [[ $crash -eq 0 && $inf -eq 0 ]]; then
	    res=`$basedir/metrics/$metric.sh $x stan`

	    if [[ $res = *"True"* ]]; then
		printf -- '-,'
	    else
		printf "*,"
	    fi
	else
	    printf -- "-,"
	fi

    fi

    if [ -e $x/pyroout* ]; then
	pyro_crash=`grep -ir "_mean" $x/pyroout* -L | wc -l`
	pyro_inf=`cat $x/pyroout* | grep -iw "inf\|nan" -l | wc -l`
	if [ $pyro_crash -eq 1 ];
	then	    
	    printf "*,"
	else
	    printf -- '-,'
	fi
	
	if [ $pyro_inf -ge 1 ];
	then	    
	    printf "*,"
	else
	    printf -- '-,'
	fi

	if [[ $pyro_crash -eq 0 && $pyro_inf -eq 0 ]]; then
        res=`$basedir/metrics/$metric.sh $x pyro`
	    if [[ $res = *"True"* ]]; then
		printf -- '-,'
	    else
		printf "*,"
	    fi	    
	else
	    printf -- "-,"
	fi
	


    fi

    if [ -e $x/edwardout* ]; then
	edward_crash=`cat $x/edwardout* | tail -3 | grep -io "[0-9]\+[\.]\?[0-9]\+" | wc -l`
	edward_inf=`cat $x/edwardout* | tail -3 | grep -iw "inf\|nan" -l | wc -l`

	if [ $edward_crash -lt 3 ];
	then	    
	    printf "*,"
	else
	    printf -- '-,'
	fi
	
	if [ $edward_inf -ge 1 ];
	then	    
	    printf "*,"
	else
	    printf -- '-,'
	fi
	if [[ $edward_crash -eq 0 && $edward_inf -eq 0 ]]; then
    	res=`$basedir/metrics/$metric.sh $x edward`
	    if [[ $res = *"True"* ]]; then
		printf -- '-'
	    else
		printf "*"
	    fi
	else
	    printf -- "-"
	fi

    fi

    printf "\n"
done

