#!/bin/bash

sacct -u jakhac --format=User,JobID,Jobname,state,time,start,end,elapsed,nodelist,MaxRSS | tail -25
#sacct --format="CPUTime,MaxRSS"
