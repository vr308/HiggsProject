#!/bin/bash

export path='/home/raid/vr308/workspace/Python/higgsDT/'

if [ $# -eq 1 ]; then 

	echo 'Running mode specified - thank you'
	export mode=$1

else 
	echo 'No argument specified by user, running in mode = test'
	echo mode=test 

fi

export LOG_FILE=higgs_DT_pipilne.log
start=`date +%s`

if [ ${mode}=test ]; then
	python results.py ${path} 2>&1 | tee ${LOG_FILE}  
else 
	python __main__.py ${mode} ${path} 2>&1 | tee ${LOG_FILE}  
fi

if(test ${PIPESTATUS[0]} -eq 0);  
then
	end=`date +%s`
	elapsed=$((end-start)) 

	echo "Probabilistic Decision-tree Algorithm finished successfully, process took ${elapsed} seconds"
	echo "Result files stored in Results/ "
	echo "Graph files stored in Graphs/ "
	echo "Please refer to logs at ${LOG_FILE}" 
	exit 0

else
	echo "Exception encountered, please check ${LOG_FILE}"
	exit 1

fi
