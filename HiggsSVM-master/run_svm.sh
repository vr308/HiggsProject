#!/bin/bash

export path='/home/raid/vr308/workspace/Python/higgsSVM/'

if [ $# -eq 2 ]; then 

	echo 'Both arguments specified - thank you'
	export train_sample_type=$1
	export mode=$2

else 

	if [ $# -eq 1 ]; then

		echo 'Usage:' 
		echo './run_svm.sh uniform_sample train' 
		echo 'or'
	  	echo './run_svm.sh choice_sample train'
		echo 'or'
		echo './run_svm.sh uniform_sample test'
		echo 'or'
		echo './run_svm.sh choice_sample test'
		echo 'Since no mode specified, running results for mode = "test" '
	
		export train_sample_type=$1
		export mode=test
else
	if [ $# -eq 0 ]; then
	
		echo 'Usage:' 
		echo './run_svm.sh uniform_sample train' 
		echo 'or'
	  	echo './run_svm.sh choice_sample train'
		echo 'or'
		echo './run_svm.sh uniform_sample test'
		echo 'or'
		echo './run_svm.sh choice_sample test'
		echo 'No input argument specified, running results for training_sample_type = "choice_sample" and mode = "test" '
	
		export train_sample_type=choice_sample
		export mode=test

	fi
	fi
fi

export LOG_FILE=higgs_SVM_pipilne_${train_sample_type}.log
start=`date +%s`

python __main__.py ${train_sample_type} ${mode} ${path} 2>&1 | tee ${LOG_FILE}  

if ( test ${PIPESTATUS[0]} -eq 0 ) ;  
then
	end=`date +%s`
	elapsed=$((end-start)) 

	echo "SVM Algorithm for ${train_sample_type} finished successfully, process took ${elapsed} seconds"
	echo "Result files stored in 'higgsSVM/Results/' folder "
	echo "Graph  files stored in 'higgsSVM/Graphs/' folder "
	echo "For logs, please refer to logs at ${LOG_FILE}" 
	exit 0

else
	echo "Exception encountered, please check ${LOG_FILE}"
	exit 1

fi
