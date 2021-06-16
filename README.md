# Subway SARS-COVID-II Simulation

This repository contains a simulation of virus transmission dynamics on a subway system. The NYC MTA's L-line and SARS-COVID-II are used as foci of the simulation's study.

***Authors: Alex Washburn, Andrew Lee, Kamil Sachryn, ChuanYao Lin, Ye Paing***

### Installation

The simulation has the follwoing requirements:

 - `python3`
 - `numpy`
 - `panda`
 - `scipy`

### Usage

The program is designed to be called via the command line.

The program has two *required* command line arguments:

 - A CSV file containing subway station information
 - A TXT file containing a train arrival schedule

This is an example of a minimal command line invocation:

```
~$ python3 subway_virus_simulation.py L-Subway-Line.csv Brooklyn-Bound.txt
```

The program has the following *optional* command line arguments:

 - `<Manhattan|Brooklyn>` *Default* `Brooklyn`
   - A mutually exclusive flag specifying in which direction the train will run.
     The direction should corresponfd to the train schedule of the required TXT file.

 - `<time>` *Default* `False`
   - A flag specifying to output the simulation runtime
     The precence of `time` is interpreted as True and the abcense is interpreted as False.

 - `<plot|no-plot>` *Default* `True`
   - A flag specifying to where or not to generate plots related to the simulation.
     The precence of `plot` is interpreted as True and the precence of `no-plot` is interpreted as False.

 - `<0123456789>` *Default* `10`
   - A positive integer specifying the number of times to replicate the simulation.

 - `<queue=0123456789>`*Default* `None`
   - A positive integer specifying the maximum queue length of passengers at all stations.
     The precence of `queue=100` is interpreted as `1000` and the absence of the option is interpreted as an infinte queue.

 - `<debug|info|warn|error>` *Default* `error`
   - A mutually exclusive flag specifying the log level. Output is logged to the `subway.log` file. 
 
 - `<anti>` *Default* `False`
   - A flag specifying to generate antithetic output. 
     The precence of `anti` is interpreted as True and the abcense is interpreted as False.
 
 - `<record>` *Default* `False`
   - A flag specifying to record the generated target random variable outputs into a `output.csv` file. 
     The precence of `record` is interpreted as True and the abcense is interpreted as False.

 - `<x%>` *Default* `1%`
   - A flag specifying the number of passengers entering the simulation as contagious.
     The number provided before the percent sign will be divided by 100.

### Example

```
~$ python3 subway_virus_simulation.py L-Subway-Line.csv Manhattan-Bound.txt Manhattan time no-plot info queue=100

Calculating the following expectation:
╭╴                                                                                   ╶────╮
│ A passenger without the virus will have the virus transmitted to them during their ride │
╰────╴                                                                                   ╶╯
Running 10 simulations of the NYC MTA's Manhattan bound L-line
Simulation complete!                              

Outcome:  0.080   ±  0.019 ( 0.060  ,  0.099  ) 95%
Runtime:  4.786s  ±  0.149 ( 4.638s ,  4.935s ) 95%
```