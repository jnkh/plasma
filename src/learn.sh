# --- the name of your job
#PBS -N learn
 
# ------------------------------------------------------------
# Job execution details
 
# --- run the job on 'n' nodes with 'y' processors per node (ppn)
#PBS -l nodes=2:ppn=2

# --- specify the memory required for a job (in mb)
#PBS -l mem=4000mb

#     hh:mm:ss (ex. 72 hours is 72:00:00)
#PBS -l walltime=48:00:00

# --- do not rerun this job if it fails
#PBS -r n

# export all my environment variables to the job
#PBS -V

#Use the gpu queue!
#PBS -q gque 

# --- define standard output and error files for the job
##PBS -o test.out
##PBS -e test.err

# ------------------------------------------------------------
# Log interesting information
#
echo " "
echo "-------------------"
echo "This is a $PBS_ENVIRONMENT job"
echo "This job was submitted to the queue: $PBS_QUEUE"
echo "The job's id is: $PBS_JOBID"
echo "-------------------"
echo "The master node of this job is: $PBS_O_HOST"

# --- count the number of processors allocated to this run 
# --- $NPROCS is required by mpirun.
set NPROCS=`wc -l < $PBS_NODEFILE`

set NNODES=`uniq $PBS_NODEFILE | wc -l`

echo "This job is using $NPROCS CPU(s) on the following $NNODES node(s):"
echo "-----------------------"
uniq $PBS_NODEFILE | sort
echo "-----------------------"

# ------------------------------------------------------------
# Setup execution variables

# cd ~/plasma/src/

# --- PBS_O_WORKDIR is the current working directory from
#     which this job was submitted using 'qsub'
echo The working directory is $PBS_O_WORKDIR 
#
#set EXEPATH=""
# ------------------------------------------------------------
# Run the program
#
# --- 'cd' to the current working directory (by default, qsub
#     starts from the home directory of the user).
#     this is useful if you have input files in the working
#     directory.
cd $PBS_O_WORKDIR
 
# --- print out the current time, run the command,
#     then print out the ending time
#
echo -n 'Started job at : ' ; date

python learn.py
 
 
echo -n 'Ended job at  : ' ; date
echo " " 
exit
