#!/bin/bash

sacct -u jakhac --format=User,JobID,Jobname,state,time,start,end,elapsed,nodelist | tail -25
