#!/bin/bash

sacct -u jakhac --state=RUNNING,PENDING --format=User,JobID,Jobname,state,time,start,end,elapsed,nodelist | tail -25
#sacct --format="CPUTime,MaxRSS"
