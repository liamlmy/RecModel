#!/bin/bash

source ../conf/common.conf
source ../conf/data.conf

exec 1>"../log/"$0"_${end_day}.log" 2>&1
set -x
set -e
if [ ${enable_fea_filt} -eq 1 ];then
	sh trans_fea_slot.sh
fi
date
if [ -z "${day_list}" ];then
	bt=`date -d"${begin_day}" +"%s"`
	et=`date -d"${end_day}" +"%s"`
	day_span=`echo "(${et}-${bt})/24*60*60" | bc`
	for((i=${day_span};i>0;i--));do
		day=`date +"%Y%m%d" -d "${i} day ago ${end_day}"`
		day_list=${day_list}""${day}" "
	done
	day=`day +"%Y%m%d" =d "${end_day}"`
	day_list=${day_list}""${day}
fi

train_hdfs_file=""
train_files="../data/train_files"
${hadoop} fs -ls ${train_hdfs_dir}/ > ${train_files}
para_num=16
i=1
for day in ${day_list};do
	echo ${day} ${train_files}
	num=`grep "${day}" ${train_files} | wc -l`
	echo ${num}
	train_hdfs_file=${train_hdfs_file}","${train_hdfs_dir}/${day}/*

	if [ ${num} -eq 0 ] || [ ${update_tfrecord_ins} -eq 1 ];then
		cd preprocess
			sh hadoop.sh ${day} &
			sleep 10s
		cd -
		((i++));
	fi
	if [ ${i} -eq ${para_num} ];then
		i=1
		wait
	fi
done
wait

train_hdfs_file=${train_hdfs_file:1}
echo train_hdfs_file, ${train_hdfs_file}
${python} -u ModelInterface.py -method auto_config -conf ${model_conf}

echo "start to train the model!"
if [ ${enable_ins} -eq 1 ];then
	HADOOP_HDFS_HOME=/usr/local/hadoop-current
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HADOOP_HDFS_HOME}/lib/native:${JAVA_HOME}/jre/lib/amd64/server
	CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
	${python} -u ModelInterface.py -method train -conf ${model_conf} -data ${train_hdfs_file}
else
	pre_process | ${python} -u ModelInterface.py -method train -conf ${model_conf}
fi
date