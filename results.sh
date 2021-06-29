stamp=$(date +"%Y-%m-%d+%T" | tr ':' '-')
iterations=1000
filePrefix="simulation-"
logSuffix="-$stamp.log"
appendSuffix="-$stamp.csv"


# Simulate increased incoming infectivity
label="infectivity"
logfile="$filePrefix-$label-$logSuffix"
appendfile="$filePrefix-$label-$appendSuffix"
for d in Brooklyn Manhattan;
do
    for p in 1 2 3 4 5;
    do
         python3 subway_virus_simulation.py $p% $iterations $d schedules/$d-Bound+X.txt L-Subway-Line.csv info no-plot quiet=true $logfile append=$appendfile
    done
done


# Simulate more train cars
label="traincars"
logfile="$filePrefix-$label-$logSuffix"
appendfile="$filePrefix-$label-$appendSuffix"
for d in Brooklyn Manhattan;
do
    for c in 8 9 10 11 12 13 14 15 16;
    do
         python3 subway_virus_simulation.py cars=$c $iterations $d schedules/$d-Bound+X.txt L-Subway-Line.csv info no-plot quiet=true $logfile append=$appendfile
    done
done         


# Simulate station limiting (fine resolution)
label="station-limit-small"
logfile="$filePrefix-$label-$logSuffix"
appendfile="$filePrefix-$label-$appendSuffix"
for d in Brooklyn Manhattan;
do
    for q in 50 60 70 80 90 100 110 120 130 140 150;
    do
         python3 subway_virus_simulation.py queue=$q $iterations $d schedules/$d-Bound+X.txt L-Subway-Line.csv info no-plot quiet=true $logfile append=$appendfile
    done
done


# Simulate station limiting (coarse resolution)
label="station-limit-large"
logfile="$filePrefix-$label-$logSuffix"
appendfile="$filePrefix-$label-$appendSuffix"
for d in Brooklyn Manhattan;
do
    for q in 200 300 400 500 600 700;
    do
         python3 subway_virus_simulation.py queue=$q $iterations $d schedules/$d-Bound+X.txt L-Subway-Line.csv info no-plot quiet=true $logfile append=$appendfile
    done
done


# Simulate different train schedules
label="schedules"
logfile="$filePrefix-$label-$logSuffix"
appendfile="$filePrefix-$label-$appendSuffix"
for d in Brooklyn Manhattan;
do
    for h in X 0 1 2 3 4 5;
    do
         python3 subway_virus_simulation.py $iterations $d schedules/$d-Bound+$h.txt L-Subway-Line.csv info no-plot quiet=true $logfile  append=$appendfile
    done
done 
