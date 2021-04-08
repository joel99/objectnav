#!/bin/bash

all_args=("$@")
echo $all_args
rest_args=("${all_args[@]:3}")
echo ${rest_args[@]}
echo "0 */6 * * * /srv/flash1/jye72/projects/embodied-recall/scripts/eval_checker.py -v $1 -s $2 -c $3 ${rest_args[@]} >> /srv/flash1/jye72/cron.log 2>&1" > crontab-fragment.txt
crontab -l | cat - crontab-fragment.txt >crontab.txt && crontab crontab.txt
