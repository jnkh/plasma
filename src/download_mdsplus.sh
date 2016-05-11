# --- the name of your job
#PBS -N download_mdsplus
 
# --- run the job on 'n' nodes with 'y' processors per node (ppn)
#PBS -l nodes=2:ppn=2

# --- specify the memory required for a job (in mb)
#PBS -l mem=4000mb

#     hh:mm:ss (ex. 72 hours is 72:00:00)
#PBS -l walltime=20:00:00

# --- do not rerun this job if it fails
#PBS -r n

# export all my environment variables to the job
#PBS -V

#Use the gpu queue!
#PBS -q dawson

# --- define standard output and error files for the job
##PBS -o download.out
##PBS -e download.err

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

# Setup execution variables

# cd ~/plasma/src/

echo The working directory is $PBS_O_WORKDIR 
#
#set EXEPATH=""

cd $PBS_O_WORKDIR

echo -n 'Started job at : ' ; date

matlab -nodisplay -nojvm -nosplash -nodesktop -r  "try, run('~/plasma/src/download_mdsplus_script.m'), catch, exit(1), end, exit(0);"
echo "matlab exit code: $?"
#PBS -r n
 
echo -n 'Ended job at  : ' ; date
echo " " 
exit
