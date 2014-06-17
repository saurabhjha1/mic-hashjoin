#!/bin/bash
## @author	Saurabh Jha <saurabh.jha.2010@gmail.com>
## @brief	Script to choose mic card and runt the code on selected mic card
## Arguments $1 = executable file name
## Arguments $2 = parameters to be passed to the code
## Example ./runOnmic.sh ./a.out "-a PRO -n 180"
HOST=mic2 
USER=jha

# copy the code to mic card, just in case it is not NFS configured. SAFE to do so.
scp $1 $HOST:/home/$USER

#run the code
ssh mic2 $1 $2
