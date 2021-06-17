import itertools
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
from   numpy.random import Generator, PCG64, SeedSequence
import os
import pandas
from   scipy import stats
import sys
import time
import subprocess
import csv


# Global variables:
transmissionRate = 1/15 # Chance of transmission per minute within 6ft of someone with the virus
controlVarResults = list() # Stores the result of Control Variable Values (SUM[amount of time passengers spent in overpacked car] / SUM[amount of total time passengers spent in a car] )
controlVarResults.clear()

class Station:

    def __init__(self, index, stationName,  inboundVirusRate, randomGenerator, arrivalRates, timeToNextStation, futureDistributions, maxQueueLength=None, startingPassengerQueue=None):
        self.index = index
        self.inboundVirusRate    = inboundVirusRate
        self.rng                 = randomGenerator
        self.arrivalRates        = arrivalRates
        self.stationName         = stationName
        self.timeToNextStation   = timeToNextStation
        self.passengerQueue      = startingPassengerQueue
        self.maxQueueLength      = maxQueueLength
        self.rebuffedPassengers  = 0
        self.futureDistributions = list()
        for (windowOpen, windowClose, dist) in futureDistributions:
            totalDist  = sum(dist)
            futureDist = list(zip(dist, range(len(dist))))
            futureDist.sort()
            futureDist.reverse()
            accumulation = 0
            for i in range(len(futureDist)):
                futureDist[i] = (futureDist[i][0] + accumulation, futureDist[i][1])
                accumulation  = futureDist[i][0]
            self.futureDistributions.append((windowOpen, windowClose, totalDist, futureDist))

        if startingPassengerQueue is None:
            self.passengerQueue = []
        else:
            self.passengerQueue = startingPassengerQueue if self.maxQueueLength == None else startingPassengerQueue[:self.maxQueueLength]


        self.MAX_STATION_INDEX = 23

    def __str__(self):
        return "Passengers/min: {} , Name: {}, mins to next station: {}\n{}".format(self.passengersPerMin,
                                                                                    self.stationName,
                                                                                    self.timeToNextStation,
                                                                                    self.futureDistribution)

    def reset(self):
        self.passengerQueue      = list()
        self.rebuffedPassengers  = 0
        
    
    def getFutureDistribution(self, clock):
        for (windowOpen, windowClose, total, future) in self.futureDistributions:
            if windowOpen <= clock and clock < windowClose:
                return (total, future)
    
    def getNextStation(self, clock):
        (totalDistribution, futureDistribution) = self.getFutureDistribution(clock)
        rv = self.rng.uniform(0, totalDistribution)
        for a,i in futureDistribution:
            if a >= rv:
                return i + 1

    #Turns out binary search is not as fast!?
    def getNextStation2(self, clock):
        (totalDistribution, futureDistribution) = self.getFutureDistribution(clock)
        rv   = self.rng.uniform(0, totalDistribution)
        arr  = futureDistribution
        high = len(arr) - 1
        low  = 0
        mid  = 0

        while low <= high:
            mid = (high + low) // 2
            if (mid == 0 and rv < arr[0][0]) or (arr[mid - 1][0] < rv and rv < arr[mid][0]):
                return arr[mid][1] + 1
            elif rv > arr[mid][0]:
                low  = mid + 1
            else: 
                high = mid - 1

    def getNextStation3(self, clock):
        totalDistribution, futureDistribution = self.getFutureDistribution(clock)
        rv   = self.rng.uniform(0, totalDistribution)
        stop = totalDistribution // 2
        low  = 0

        # Linear probing until we have accounted for half the distribution space
        # This will probably not be more than 3 cells
        for a,i in futureDistribution:
            if a >= rv:
                return i + 1
            low += 1
            if a >= stop:
                break

        # After the initial linear probing,
        # switch to binary search for the remaining distribution space
        arr  = futureDistribution
        high = len(arr) - 1
        mid  = 0
        while low <= high:
            mid = (high + low) // 2
            if arr[mid - 1][0] < rv and rv < arr[mid][0]:
                return arr[mid][1] + 1
            elif rv > arr[mid][0]:
                low  = mid + 1
            else: 
                high = mid - 1
    
    #Pop passengers waiting at station
    def LoadPassengers(self):
        passengers = self.passengerQueue.copy()
        self.passengerQueue.clear()
        return passengers

    #Add passengers to station
    def QueuePassengers(self, passengers):
        for p in passengers:
            self.passengerQueue.append(p)

    #returns passengersPerMin Passengers
    def GeneratePassengers(self, startTime, elapsingTime=1, antithetic=False):
        endTime = startTime + elapsingTime
        arrivalIntensity = 0
        it = iter(self.arrivalRates)        
        for (windowOpen, windowClose, rate) in it:
            if (windowOpen <= startTime and startTime < windowClose):
                if (endTime < windowClose):
                    arrivalIntensity = rate * elapsingTime
                else:
                    (x, y, rate2)    = next(it)
                    overflow         = (endTime - windowClose)
                    arrivalIntensity = rate * (elapsingTime - overflow) + rate2 * overflow

        queueSpace  = sys.maxsize if self.maxQueueLength == None else self.maxQueueLength - len(self.passengerQueue)
        passengers  = list()
        rv          = self.rng.poisson(arrivalIntensity)
        (numArrivals, rebuffed) = (min(queueSpace, rv), max(0, rv-queueSpace))
        self.rebuffedPassengers += rebuffed
        for i in range(numArrivals):
            stationsUntilExit = self.getNextStation3(startTime)
            startsWithVirus = False

            ################################### DREW's VR EXPRIMENT ################################################
            if(antithetic):
                rv = 1 - self.rng.uniform() # RV for incomming virus rate
            else:
                rv = self.rng.uniform() 
            #################################### END OF EXPERIMENT #################################################
            if rv < self.inboundVirusRate:
                startsWithVirus = True
                debug("VIRUS", self.stationName)
            passengers.append(Passenger(stationsUntilExit, startsWithVirus))

        self.QueuePassengers(passengers)
        return numArrivals


class Train:

    def __init__(self, uniqueID, carCount, maximumCapacity, safeCapacity):
        self.identifier        = uniqueID
        self.cars              = list()
        self.minsToNextStation = 0

        for i in range(carCount):
          self.cars.append(Car_Hex(maximumCapacity))
#          self.cars.append(Car(maximumCapacity, safeCapacity))
        
    #Add 1 passenger to a safe zone, returns false if train car is at max capacity
    def AddPassenger(self, passenger):
        bestCar = self.cars[0]
        for car in self.cars:
            if car.GetOccupancy() < bestCar.GetOccupancy():
                bestCar = car
        return bestCar.AddPassenger(passenger)

    #make passengers depart if their stop counter is 0
    def ArriveAtStation(self):
        passengersDeparting : Passenger = list()
        for car in self.cars:
            passengersDeparting += car.ArriveAtStation()
        return passengersDeparting

    #tick mins
    def DecrementMinsToNextStation(self):
        self.minsToNextStation -= 1

    #Update everything
    def Tick(self, time=1):
        for car in self.cars:
            car.Tick(time)

    def showTrain(self):
        print("Train #", self.identifier)
        count = 0
        for car in self.cars:
            count += 1
            print("  Car #", count)
            print(car)


class Car_Hex:
    
    def __init__(self, maxCapacity):
        # Number of hexes *between* two cells for them to be considered socially distanced
        self.socialDistance = 4

        # Dimensions of the hexagonal grid
        self.dimensions     = (8, 46)
        
        # List of 24 pre-computed "safe spaces" in the train car where,
        # if passengers only occupy these spaces,
        # up to 24 passengers can social distance while in transit.
        self.safeSpaces     = [ (0,5), (0,13), (0,21), (0,29), (0,37), (0,45),
                                (2,1), (2, 9), (2,17), (2,25), (2,33), (2,41),
                                (5,4), (5,12), (5,20), (5,28), (5,36), (5,44), 
                                (7,0), (7, 8), (7,16), (7,24), (7,32), (7,40)]

        # An 8 x 46 hexagonal grid of possible passenger locations.
        # Uses the "odd row" horizontal layout 2D coordinate system described here:
        #  > https://www.redblobgames.com/grids/hexagons/#coordinates
        self.spaces         = np.full(self.dimensions, None)

        # Maximum number of passengers that can occupy the train car,
        # even if the hexagonal grid is not full.
        # However, the maximum capacity cannot exceed the hexagonal grid capactiy.
        self.maxCapacity    = min(maxCapacity, math.floor(self.dimensions[0] * (self.dimensions[1] - 0.5)))

        # A list of indices of all passnegers on the train car (including "safe spaces").
        self.occupiedSpaces = set()

        # A list of indices in which passengers with the virus are located.
        # A subset of occupiedSpaces
        self.virusSpaces    = set()

    def __str__(self):
        # Nicely render the hexagonal grid of passengers
        output = ""
        for i in range(len(self.spaces)):
            x = 0
            if i % 2 != 0:
                output += "   "
                x = 1
            else:
                x = 0
            for j in range(len(self.spaces[i]) - x):
                p = self.spaces[i][j]
                if p == None:
                    output += "[   ] "
                elif p.HasVirus():
                    output += "[ V ] "
                else:
                    output += "[ P ] "
            output += '\n'
        return output

    # Add 1 passenger to the car, returns false if train car is at max capacity
    def AddPassenger(self, passenger):
        occupancy = self.GetOccupancy()

        # If the train car is at capacity, the passenger does not board.
        if occupancy >= self.maxCapacity:
            return False

        e = None
        # If there is an open "safe space," the passenger goes there.
        if occupancy < len(self.safeSpaces):
            e = self.safeSpaces[occupancy]

        # Otherwise, we select an empty cell uniformly at random.
        else:
            while True: # Risks "live-lock," but whatcha gonna do?
                # Pick row uniformly at random
                i = np.random.randint(0, self.dimensions[0])
                # If the the row is even, the row is full length
                # If the row is odd, the row is one less the full length
                j = np.random.randint(0, self.dimensions[1] if i % 2 == 0 else self.dimensions[1] - 1)
                e = (i,j)
                # If there is no passenger at the selected space, were' done!
                # Otherwise we select a random cell again.
                if self.spaces[e] == None:
                    break
        
        self.spaces[e] = passenger
        self.occupiedSpaces.add(e)
        if passenger.HasVirus():
            self.virusSpaces.add(e)                        
        return True
        

    # Make passengers depart if their stop counter is 0
    def ArriveAtStation(self):
        startingOccupancy = self.GetOccupancy()

        #Determine the indicies of passenger who are departing and who will continue to ride
        departing, riding = set(), set()
        for e in self.occupiedSpaces:
            passenger = self.spaces[e]
            passenger.DecrementStops()
            (departing if passenger.GetStopsUntilDisembark() <= 0 else riding).add(e)

        # For every passenger that is departing
        # Remove the passenger from the hexagonal grid,
        # and the occupancy and virus spaces lists.
        # Then add the passenegr to the list of departures.
        passengersDepartures : Passenger = list()
        for e in departing:
            passenger = self.spaces[e]
            if passenger.HasVirus():
                self.virusSpaces.discard(e)
            self.spaces[e] = None
            passengersDepartures.append(passenger)
    
        # If the number of passengers in the train car started above the number of safe spaces
        # and after disembarking there are fewer than passengers than the number of safe spaces
        # then we rearrange the passengers to occupy only safe spaces.
        startingOccupancy = self.GetOccupancy()
        currentOccupancy  = len(riding)
        if startingOccupancy > len(self.safeSpaces) and currentOccupancy <= len(self.safeSpaces):
            openSafeSpaces = set(self.safeSpaces)
            unsafeSpaces   = set()
            finalSpaces    = set()
            
            for e in riding:
                if e in openSafeSpaces:
                    openSafeSpaces.discard(e)
                    finalSpaces.add(e)
                else:
                    unsafeSpaces.add(e)

            for (u,s) in list(zip(unsafeSpaces, openSafeSpaces)):
                self.spaces[s] = self.spaces[u]
                self.spaces[u] = None
                finalSpaces.add(s)

            riding = finalSpaces

        self.occupiedSpaces = riding
        
        return passengersDepartures


    def GetOccupancy(self):
        return len(self.occupiedSpaces)
    

    def __GetSocialDistanceRange(self, i, j):
        # Top half (plus center row)
        for x in range(0 - self.socialDistance, 1):
            a = i + x
            if 0 <= a and a < self.dimensions[0]:
                for y in range(0 - self.socialDistance + abs(math.ceil(x/2)), self.socialDistance + 1 - abs(math.floor(x/2))) if i%2 == 0 else range(0 - self.socialDistance + abs(math.floor(x/2)), self.socialDistance + 1 - abs(math.ceil(x/2))):
                    b = j + y
                    if 0 <= b and b < self.dimensions[1] and (a,b) != (i,j):
                        yield (a, b)
    
        # Lower half
        for x in range(1, self.socialDistance + 1):
            a = i + x
            if 0 <= a and a < self.dimensions[0]:
                for y in range(0 - self.socialDistance + abs(math.floor(x/2)), self.socialDistance + 1 - abs(math.ceil(x/2))) if i%2 == 0 else range(0 - self.socialDistance + abs(math.ceil(x/2)), self.socialDistance + 1 - abs(math.floor(x/2))):
                    b = j + y
                    if 0 <= b and b < self.dimensions[1]:
                        yield (a, b)
                    
    
    # Do everything needed to simulate an interval of time passing within the train car
    def Tick(self, time=1):
        if time < 1:
            return

        # First, increment all passenger ride times
        for e in self.occupiedSpaces:
            self.spaces[e].IncrementRideTime(time)
            # Increment the time passenger spent being inside an overpacked car
            if self.GetOccupancy() > len(self.safeSpaces):
                self.spaces[e].IncrementOverpackedTime(time)

        # Every passenger which has the virus transmitted to them during the time interval.
        newViralPassengers = set()
        viralExposureTasks = list(zip(self.virusSpaces, itertools.count(time,0)))

        # Second, handle virus exposure and transmission
        while viralExposureTasks:
            currentTasks       = viralExposureTasks
            viralExposureTasks = list()

            # For all the tasks we knew about at the start of the loop,
            # Process each task.
            # Each task processed may add one or more tasks.
            for (v,t) in currentTasks:
                # Get all the cells within the social distancing radius
                for e in self.__GetSocialDistanceRange(v[0],v[1]):
                    if e in self.occupiedSpaces:
                        passenger = self.spaces[e]
                        # If there is a passenger with the radius,
                        # Increase the passenger's exposure time appropriately
                        extraTime = passenger.AddExposureTime(t)
                        # If there was not a negative number of minutes
                        # exceeding the passenger's time until transmission
                        # then the passenger has had the virus transmitted to them
                        # and we must add their indicies to the list of virus passengers.
                        if extraTime >= 0:
                            newViralPassengers.add(e)
                        # If the passenger had a *positive* number of minutes
                        # exceeding their time until transmission,
                        # then they will expose surrounding passengers
                        # for the duration of the excess time.
                        # We add this passenger as a new task to handle
                        # during the next main loop of the function.
                        if extraTime >  0:
                            viralExposureTasks.append((e,t))

        self.virusSpaces = self.virusSpaces.union(newViralPassengers)

 
class Car:

    def __init__(self, maximumCapacity, safeCapacity):
        self.maximumCapacity = maximumCapacity
        self.safeCapacity = safeCapacity
        self.spaces = list() # [[None for i in range(int(maximumCapacity/safeCapacity))] for j in range(safeCapacity)]
        self.occupancy = 0

        for i in range(safeCapacity):
            self.spaces.append(list())

    def __str__(self):
        output = ""
        for i in self.spaces:
            output += str(i) + '\n'
        return output

    #Add 1 passenger to a safe zone, returns false if train car is at max capacity
    def AddPassenger(self, passenger):
        if self.occupancy >= self.maximumCapacity:
            return False
        bestSafeZone = self.__GetBestSafeZone()
        if bestSafeZone == -1:
            return False
        else:
            self.__AddPassengerToSafeZone(passenger, bestSafeZone)
            self.occupancy += 1
            return True

    #Add passenger to safe zone at first available index
    def __AddPassengerToSafeZone(self, passenger, index):
        self.spaces[index].append(passenger)

    def GetOccupancy(self):
        return self.occupancy
    
    #Determines safezone with least people
    def __GetBestSafeZone(self):
        bestIndex = -1
        bestPeeps = self.maximumCapacity + 1
        for i in range(len(self.spaces)):
            peeps = len(self.spaces[i])
            if peeps < bestPeeps:
                bestIndex = i
                bestPeeps = peeps
        return bestIndex

    #make passengers depart if their stop counter is 0
    def ArriveAtStation(self):
        passengersDeparting : Passenger = list()
        for i in range(len(self.spaces)):
            departing, riding = [], []
            for passenger in self.spaces[i]:
                passenger.DecrementStops()
                (departing if passenger.GetStopsUntilDisembark() <= 0 else riding).append(passenger)
            self.spaces[i] = riding
            passengersDeparting += departing
            self.occupancy -= len(departing)
        return passengersDeparting

    #Do everything needed every minute
    #Incremenets passengers vars
    def Tick(self, time=1):
        if time < 1:
            return
        for i in range(len(self.spaces)):
            numberWithVirus = 0
            for passenger in self.spaces[i]:
                if passenger.HasVirus():
                    numberWithVirus += 1
            for passenger in self.spaces[i]:
                passenger.IncrementRideTime(time)
                if numberWithVirus > 0:
                    if not passenger.HasVirus():
                        passenger.AddExposureTime( numberWithVirus    * time)
                    else:
                        passenger.AddExposureTime((numberWithVirus-1) * time)



class Passenger():
    def __init__(self, stopsUntillDisembark, startWithVirus):
        self.stopsUntillDisembark  = stopsUntillDisembark
        self.hasVirus              = startWithVirus
        self.rideTime              = 0
        self.exposureTime          = 0
        self.overpackedTime        = 0
        self.startedWithVirus      = startWithVirus
        self.novelTransmissionTime = 0
        self.timeUntilTransmission = None

    def __str__(self):
        return "stops to go: {}, Virus: {}, ride time: {}, exposure time: {}".format(self.stopsUntillDisembark,
                                                                                     self.hasVirus,
                                                                                     self.rideTime,
                                                                                     self.exposureTime)

    def DecrementStops(self):
        self.stopsUntillDisembark -= 1

    def GetExposureTime(self):
        return self.exposureTime
    
    def GetOverpackedTime(self):
        return self.overpackedTime

    def HasVirus(self):
        return self.hasVirus

    def GetStopsUntilDisembark(self):
        return self.stopsUntillDisembark

    def GetRideTime(self):
        return self.rideTime

    # Adds time a passenger spent being in overpacked car to passenger
    def IncrementOverpackedTime(self, minutes):
        self.overpackedTime += minutes

    # Adds exposure time to passenger
    def AddExposureTime(self, minutes):
        # It doesn't matter if you already have the virus,
        # We track the total exposure time of *ALL* passengers
        self.exposureTime += minutes

        # But if you already have the virus, we don't need to worry about the rest.
        if self.hasVirus:
            return -1

        # Lazily sample the geometric distribution *only*
        # the first time the passenger is exposed to the virus.
        # Save some work since most passenegrs are never exposed,
        # and sampling the geometric distribution is expensive.
        if self.timeUntilTransmission == None:
            self.timeUntilTransmission = np.random.geometric(p=transmissionRate)

        # Calculate how many minutes are "left over"
        # after having the virus transmitted to the passenger.
        # A negative number represents that transmission has not yet occured.
        exceededTime = self.exposureTime - self.timeUntilTransmission
        if exceededTime >= 0:
            debug("TRANS")
            self.hasVirus = True
            # TODO remove this
            self.novelTransmissionTime = self.rideTime

        # Return the number of "left over" minutes after transmission occured.
        # We need to make sure to calculate other passenger's exposure
        # for this duration of time if the value is positive.
        return exceededTime

    def IncrementRideTime(self, time=1):
        self.rideTime += time

    def StartedWithVirus(self):
        return self.startedWithVirus

    def IsNovelTransmission(self):
        return self.hasVirus and not self.startedWithVirus

    # Number of minutes that the passenger was exposed
    # until the virus was transmitted to them
    def GetNovelTransmissionTime(self):
        return self.timeUntilTransmission


def debug(label, *argv):
    width  = 5
    tag    = label[:width].upper()
    prefix = ' ' *          ((width - len(tag))//2)
    suffix = ' ' * math.ceil((width - len(tag))/ 2)
    padded = "[" + prefix + tag + suffix + "]: "
    logging.debug(padded + ' '.join(map(str, list(argv))))

    
# Use a more efficient method following Prof. Vazquez-Abad's "Ghost Bus" model.
def Simulation_Retrospective(trainSchedule, subwayLine, antithetic=False, carsPerTrain=8, carSafeCapacity=24, carMaxCapacity=258):

    clock              = 0
    trainIndex         = 0
    departedPassengers = list()
    trainsEnroute      = list()
    #Initialize stations
    for i in range(len(subwayLine)):
        delta = subwayLine[i].timeToNextStation
        for j in range(i+1,len(subwayLine) - 1):
            subwayLine[j].GeneratePassengers(clock, delta, antithetic)

    while trainSchedule:

        # Determine how much time elapsed between the previous train and the current train 
        arrival = trainSchedule.pop(0)
        elapsed = arrival - clock

        # Simulate passengers arriving at all stations for the elapsed time between trains
        for station in subwayLine[:-1]:
            station.GeneratePassengers(clock, elapsed, antithetic)

        # Advance the simulation clock to the arrival fo the next train
        clock += elapsed
        debug("TICK", "(", clock, ")")

        # Create the next train
        trainIndex += 1
        train = Train(trainIndex, 8, carMaxCapacity, carSafeCapacity)
        debug("ADD", "(", clock,"):", trainIndex)

        # Move the train throught the subway line
        for station in subwayLine:
            debug("STATN","(", clock, "):", train.identifier, station.stationName)

            #Get passengers off trains who arrived at their stop, add to list for later
            departingPassengers = train.ArriveAtStation()
            for p in departingPassengers:
                departedPassengers.append(p)

            #get passengers waiting to board train at station
            passengersToLoad = station.LoadPassengers()
            for p in passengersToLoad:
                if train.AddPassenger(p): #if passenger was succesfully added
                   passengersToLoad.remove(p)
                else: #train is full, exit loop
                    break

            #if train filled before station emptied, add passengers back into station queue
            if len(passengersToLoad) > 0:
                station.QueuePassengers(passengersToLoad)

            #train.showTrain()
                
            # Tick trains (updates new values for passengers)
            train.Tick(station.timeToNextStation)

    rebuffedTally = 0
    for station in subwayLine:
        rebuffedTally += station.rebuffedPassengers
    return (departedPassengers, rebuffedTally)


def generateControlVariable(departedPassengers):
    #Main loop ended, show control variable data gathered.
    totalOverpackedTime        = 0
    totalRideTime              = 0

    for passenger in departedPassengers:
        totalRideTime += passenger.GetRideTime()
        totalOverpackedTime += passenger.GetOverpackedTime()

    return totalOverpackedTime , totalRideTime 

def generateStatistics(departedPassengers, rebuffedTally):
    #Main loop ended, show data gathered.
    startVirusPassengers = 0
    novelVirusPassengers = 0
    exposedPassengers    = 0
    safePassengers       = 0
    totalRideTime        = 0
    totalExposureTime    = 0
    for passenger in departedPassengers:
        totalRideTime += passenger.GetRideTime()

        if passenger.StartedWithVirus():
            startVirusPassengers += 1
        elif passenger.IsNovelTransmission():
            novelVirusPassengers += 1
        elif passenger.GetExposureTime() > 0:
            exposedPassengers += 1
            totalExposureTime += passenger.GetExposureTime()
        else:
            safePassengers += 1


    totalPassengers = startVirusPassengers + novelVirusPassengers + exposedPassengers + safePassengers
    simulationStatistics = ''.join([ '\nPassengers (Start Virus): ', str(startVirusPassengers)
                                   , '\nPassengers (Novel Virus): ', str(novelVirusPassengers)
                                   , '\nPassengers (Total Virus): ', str(startVirusPassengers + novelVirusPassengers)
                                   , '\nPassengers (Exposed    ): ', str(   exposedPassengers)
                                   , '\nPassengers (Safe       ): ', str(      safePassengers)
                                   , '\nPassengers (Total      ): ', str(totalPassengers)
                                   , '\nPassengers (Rebuffed   ): ', str(rebuffedTally), ' (' + str(round (100*(rebuffedTally / totalPassengers), 4)) + '%)'
                                   , '\n\nTransit  Time: ', str(totalRideTime)
                                   ,   '\nExposure Time: ', str(totalExposureTime)
                                   , '\n\nExpected exposure time (per passenger)', str(0 if len(departedPassengers) == 0 else totalExposureTime / len(departedPassengers))
                                   ,   '\nExpected exposure time (per minute   )', str(0 if totalRideTime == 0 else totalExposureTime / totalRideTime)
                                   , '\n\nExpected novel transmission (per "healthy" passenger)', str(0 if len(departedPassengers) == 0 else novelVirusPassengers / (len(departedPassengers) - startVirusPassengers))
    ])

    logging.info(simulationStatistics)


def generatePlot(departeds, direction):
    N = len(departeds)

    # Extract simulation results/transmission time for each simulation
    simulation_results = []
    novel_transmission_sim_results = []
    novel_transmission_mins = [0]

    for departedPassengers in departeds:
        results = []
        transmission_results = []
        for passenger in departedPassengers:
            if passenger.GetExposureTime() > 0:
                results.append(passenger.GetExposureTime())
            if passenger.IsNovelTransmission():
                novelTransmissionTime = passenger.GetNovelTransmissionTime()
                transmission_results.append(novelTransmissionTime)
                if novelTransmissionTime not in novel_transmission_mins:
                    novel_transmission_mins.append(novelTransmissionTime)

        simulation_results.append(results)
        novel_transmission_sim_results.append(transmission_results)

    data = dict()
    minute_list = []

    # Grab unique minute "buckets" from all simulations
    for simulation in simulation_results:
        for minute in simulation:
            if minute not in minute_list:
                        minute_list.append(minute)

    # Sort these minute buckets and turn them into a dictionary
    minute_list.sort()
    for m in minute_list:
        # Key m = minute bucket
        # Tuple (Max, Min, Total)
        data[m] = (None, None, None)

    # For each simulation, we'll tally the results into our dictionary
    for simulation in simulation_results:
        for d in data:
            max, min, total = data[d]
            sim_count = simulation.count(d)

            max = sim_count if max is None or sim_count > max else max
            min = sim_count if min is None or sim_count < min else min
            total = sim_count if total is None else sim_count + total

            data[d] = (max, min, total)


    # Print/plot the results:
    bin_max = 0
    hist_max = []
    hist_min = []
    hist_median = []

#    print("\n\nMinute => Max Exposure Count, Min Exposure Count, Total Exposure Count")
    for d in data:
        bin_max = d if d > bin_max else bin_max
        max, min, total = data[d]
#        print(str(d) + " => " + str(max) + ", " + str(min) + ", " + str(total))

        for i in range(max):
            hist_max.append(d-1)

        for i in range(min):
            hist_min.append(d-1)

        median_count = max - int((max - min)/2);
        for i in range(median_count):
            hist_median.append(d-1)

    bins = list(range(bin_max+1))



    # Do the same for transmission simulation:
    transmissions_data = dict()
    novel_transmission_mins.sort()

    for m in novel_transmission_mins:
        # Max transmissions count, min min transmissions count at minute m
        transmissions_data[m] = (None, None)

    # For each simulation, we'll tally the results into our dictionary
    for simulation in novel_transmission_sim_results:
        for d in transmissions_data:
            max, min = transmissions_data[d]
            sim_count = simulation.count(d)

            max = sim_count if max is None or sim_count > max else max
            min = sim_count if min is None or sim_count < min else min

            transmissions_data[d] = (max, min)

    # Print/plot the results:
    transmissions_bin_max = 0
    transmissions_hist_max = []
    transmissions_hist_min = []

    for d in transmissions_data:
        transmissions_bin_max = d if d > transmissions_bin_max else transmissions_bin_max
        max, min = transmissions_data[d]

        for i in range(max):
            transmissions_hist_max.append(d-1)

        for i in range(min):
            transmissions_hist_min.append(d-1)

    novel_transmission_bins = list(range(transmissions_bin_max+1))
    # 
    # Render the plot
    #

    fig, ax = plt.subplots(2)

    # Exposure Graph
    ax[0].set_title('Max/Min Exposure Results For ' + str(N) + " Iterations [" + direction + " Bound]")
    ax[0].set_xlabel('Minutes Exposed')
    ax[0].set_ylabel('Max/Min Number of Passengers')
    ax[0].hist(hist_max, bins, color='red', edgecolor='red', linewidth=1)
    ax[0].hist(hist_min, bins, color='blue', edgecolor='blue', linewidth=1)

    # Transmission Graph
    ax[1].set_title('Max/Min Novel Transmission Results For ' + str(N) + " Iterations [" + direction + "Bound]")
    ax[1].set_xlabel('Max/Min Transmissions At Minutes')
    ax[1].set_ylabel('Number of Passengers')
    ax[1].hist(transmissions_hist_max, novel_transmission_bins, color='red', edgecolor='red', linewidth=1)
    ax[1].hist(transmissions_hist_min, novel_transmission_bins, color='blue', edgecolor='blue', linewidth=1)

    # Uncomment below to show every minute tick.
    # plt.xticks(bins)

    plt.show()

def replicateSimulation(subwayLine, trainSchedule, n=10, direction="Brooklyn", plotStats=True, timeResults=True, antithetic=False):
    if direction.lower() != "brooklyn" and direction.lower() != "manhattan":
        print("Unrecognized direction! ", direction, "\nExpecting one of:\n  - Brooklyn\n  - Manhattan")
        return

    print("\nCalculating the following expectation:")
    print("╭╴                                                                                   ╶────╮")
    print("│ A passenger without the virus will have the virus transmitted to them during their ride │")
    print("╰────╴                                                                                   ╶╯")
    print("Running", n, "simulations of the NYC MTA's", direction, "bound L-line")

    departeds = []
    rebuffeds = []
    timings   = []
    progressLimit = 50
    for i in range(n):
        sys.stdout.write('.')
        sys.stdout.flush()
        tStart = time.time()
        (passengers, rebuffed) = Simulation_Retrospective(trainSchedule.copy(), subwayLine.copy(), antithetic)
        tEnd   = time.time()
        if i % progressLimit == 0:
            sys.stdout.write('\b'*progressLimit + ' ' * progressLimit + '\b'*progressLimit)
            sys.stdout.flush()
        
        departeds.append(passengers)
        rebuffeds.append(rebuffed)
        timings.append(tEnd - tStart)
        for station in subwayLine:
            station.reset()

    sys.stdout.write('\b'*progressLimit + ' ' * progressLimit + '\b'*progressLimit)
    
    print("Simulation complete!\n")

    controlVarResults.clear()

    for i,j in zip(departeds, rebuffeds):
        generateStatistics(i,j)
        controlVarResults.append(generateControlVariable(i))

    # Get the simulation outcome
    outcomes = []
    for simulation_result in departeds:
        outcomes.append(calculateRV(simulation_result))
    outcomeStr = confidenceIntervalString(outcomes)
    print("Outcome:", outcomeStr)
  

    # Conditionally print the runtime results
    if timeResults:
        timingStr = confidenceIntervalString(timings, 's')
        print("Runtime:", timingStr)

    if plotStats:
        generatePlot(departeds, direction)

    return outcomes


def confidenceIntervalString(observations, unit='', confidence=95):
    # Define some constants required for pretty rendering
    unitCharaters  = 2
    unitTruncated  = unit[:unitCharaters]
    u              = unitTruncated + (' ' * (unitCharaters - len(unitTruncated)))

    # Calculate the mean of the observations
    mean           = sum(observations) / len(observations)
    
    # If there's only one run of the simulation,
    # then there is no confidence interval for the mean runtime result
    confidenceStr  = ""
    n              = len(observations)
    if n > 1:
        totalDeviations = 0
        for x in observations:
            y = max(mean,x) - min(mean,x)
            totalDeviations += y*y
        s = math.sqrt(totalDeviations / (n - 1))
        t = stats.t.ppf(1 - ((100-confidence)/2/100), n - 1)
        v = (t*s) / math.sqrt(n)
        plusMinus  = nicelyRenderDecimal(              v )
        lowerBound = nicelyRenderDecimal(max(0, mean - v))
        upperBound = nicelyRenderDecimal(       mean + v )
        confidenceStr = " ± " + plusMinus + " (" + lowerBound + u + ", " + upperBound + u + ") 95%"

    # Return a pretty rendering of the mean and confidence interval of the observations
    return nicelyRenderDecimal(mean) + u + confidenceStr


def nicelyRenderDecimal(value, characteristic=2, mantissa=5):
    roundedStr        = str(round(value, mantissa))
    x                 = roundedStr.split('.', 1)
    characteristicStr = x[0]
    mantissaStr       = x[1] if len(x) > 1 else ""
    prefix            = ' ' * (characteristic - len(characteristicStr))
    suffix            = '0' * (mantissa       - len(      mantissaStr))
    return prefix + characteristicStr + '.' + mantissaStr + suffix 


def calculateRV(departedPassengers):
    N = len(departedPassengers)
    didNotStartWithVirus = 0
    novelTransmissions   = 0
    
    for passenger in departedPassengers:
        if not passenger.StartedWithVirus():
            didNotStartWithVirus += 1
            if passenger.IsNovelTransmission():
                novelTransmissions += 1

    return novelTransmissions / didNotStartWithVirus


def parseSubwayLineFile(subwayFile, direction, inboundVirusRate, maxQueueLength, randomSeed):
    stations = pandas.read_csv(subwayFile)
    manhattanBound = direction == "Manhattan"

    # Create a set of all station names
    stationNames = set()
    for name in stations['Station']:
        stationNames.add(name)

    # Find the starting station by determining
    # which station's "previous station" is not in the set of station names
    startingStation = -1
    for i in stations.index:
        if  (manhattanBound and not stations[ 'Brooklyn bound station'][i] in stationNames) or (not manhattanBound and not stations['Manhattan bound station'][i] in stationNames):
            startingStation = i
            break

    # Deterimine the order of row indices corresponding to the subway line direction
    # This allows for the CSV file data to be treated as a "linked-list-like" input
    # There is no assumption about input row ordering,
    # any permutation of input rows will create the same subway line
    currentStation = startingStation
    stationOrdering = list()
    while True:
        stationOrdering.append(currentStation)
        nextStationName = stations['Manhattan bound station'][currentStation] if manhattanBound else stations['Brooklyn bound station'][currentStation]
        nextStation = -1
        for i in stations.index:
            if stations['Station'][i] == nextStationName:
                nextStation = i
                break
        if nextStation < 0:
            break
        else:
            currentStation = nextStation

    arrivalTimeWindows = list()
    for col in stations.columns:
        if col.startswith('Arrival'):
            strRange = col[7:].split('-',2)
            strBegin = strRange[0].strip()
            strEnd   = strRange[1].strip()
            begin    = parseTimestamp(strBegin)
            end      = parseTimestamp(strEnd)
            window   = (col, begin, end)
            arrivalTimeWindows.append(window)
            
    departureTimeWindows = list()
    for col in stations.columns:
        if col.startswith('Departure'):
            strRange = col[9:].split('-',2)
            strBegin = strRange[0].strip()
            strEnd   = strRange[1].strip()
            begin    = parseTimestamp(strBegin)
            end      = parseTimestamp(strEnd)
            window   = (col, begin, end)
            departureTimeWindows.append(window)
            
    index = 0
    subwayLine = list()
    gens = list()
    sg   = SeedSequence(randomSeed)
    for s in sg.spawn(len(stationOrdering)):
        gens.append(Generator(PCG64(s)))

    for i in range(len(stationOrdering)):
        k = stationOrdering[i]
        timeToNext = stations['Manhattan bound time'][k] if manhattanBound else stations['Brooklyn bound time'][k]
        timeToNext = 0 if math.isnan(timeToNext) else int(timeToNext)

        arrivalRates = list()
        for (col, begin, end) in arrivalTimeWindows:
             arrivalRates.append((begin, end, stations[col][k]))
        arrivalRates.sort()

        dists = list()
        for (col, begin, end) in departureTimeWindows:
            departRates = list()
            for j in range(i+1,len(stationOrdering)):
               departRates.append(stations[col][j])
            departRates.sort()
            dists.append((begin, end, departRates))

        subwayLine.append(Station(index, stations['Station'][k], inboundVirusRate, gens[k], arrivalRates, timeToNext, dists, maxQueueLength))
        index += 1

    return subwayLine


def parseTrainScheduleFile(scheduleFile):
    arrivalTimes = set()
    with open(scheduleFile, 'r') as file:
        schedule = file.read().split()
        for timestamp in schedule:
            arrivalTimes.add(parseTimestamp(timestamp))
        file.close()

    arrivalTimes = list(arrivalTimes)
    arrivalTimes.sort()
    return arrivalTimes


def parseTimestamp(timestamp):
    cut     = timestamp.split(':',2)
    hours   = int(cut[0])
    minutes = int(cut[1])
    return hours*60 + minutes    


def main():

    # Set random seed for reproducability
    # Using the first 9 digits of ϕ (phi)
    # as a "nothing up our sleeve" choice.
    np.random.seed(618033988)

    largs = list()
    for arg in sys.argv:
        largs.append(arg.lower())

    stationFile = None
    for i in range(len(largs)):
        if largs[i].endswith('.csv'):
            stationFile = sys.argv[i]
            break

    if stationFile == None:
        print("Missing required command line option:\n  CSV file of subway line station information")
        os._exit(os.EX_DATAERR)

    scheduleFile = None
    for i in range(len(largs)):
        if largs[i].endswith('.txt'):
            scheduleFile = sys.argv[i]
            break

    if scheduleFile == None:
        print("Missing required command line option:\n  TXT file of train schedule information")
        os._exit(os.EX_DATAERR)

    logFile = "subway.log"
    for i in range(len(largs)):
        if largs[i].endswith('.log'):
            logFile = sys.argv[i]
            break

    record = False
    for i in range(len(largs)):
        if largs[i] == 'record':
            record = True

    antithetic = False
    for i in range(len(largs)):
        if largs[i] == 'anti':
            antithetic = True

    direction = "Brooklyn" # default
    if "brooklyn" in largs:
        direction = "Brooklyn"
    elif "manhattan" in largs:
        direction = "Manhattan"

    iterations = 10
    for arg in largs:
        if arg.isdigit():
            iterations = int(arg)
            break

    plotStats = True
    if "plot" in largs:
         plotStats = True
    elif "no-plot" in largs:
         plotStats = False

    logLevel = level=logging.ERROR
    for arg in largs:
        if arg == "debug":
            logLevel = logging.DEBUG
            break
        elif arg == "log" or arg == "info":
            logLevel = logging.INFO
            break
        elif arg == "warn":
            logLevel = logging.WARN
            break

    timeSimulation = False
    for arg in largs:
        if arg == "time":
            timeSimulation = True
            break

    maxQueueLength = None
    for i in range(len(largs)):
        arg = largs[i]
        if arg.startswith('queue='):
            queueArg = arg[6:]
            if queueArg.isdigit():
                maxQueueLength = int(queueArg)
                maxQueueLength = None if maxQueueLength < 1 else maxQueueLength
                break
            else:
                print("Input error in command line option:\n ", sys.argv[i], "does not specify an integer")
                os._exit(os.EX_DATAERR)

    inboundVirusRate = 0.01
    for i in range(len(largs)):
        if largs[i].endswith('%'):
            try:
                inboundVirusRate = float(sys.argv[i][:-1]) / 100
                break
            except ValueError:
                print("Input error in command line option:\n ", sys.argv[i], "does not specify an number")
                os._exit(os.EX_DATAERR)

    # Set random seed for reproducability
    # Using the first 9 digits of π (pi)
    # as a "nothing up our sleeve" choice.
    randomSeed = 141592653
    for i in range(len(largs)):
        if largs[i].startswith('@'):
            try:
                randomSeed = int(sys.argv[i][1:])
                break
            except ValueError:
                print("Input error in command line option:\n ", sys.argv[i], "does not specify an number for the random seed")
                os._exit(os.EX_DATAERR)

    outFile = None
    for i in range(len(largs)):
        arg = largs[i]
        if arg.startswith('output='):
            outFile = arg[7:]

    print(sys.argv)

    subwayLine    = parseSubwayLineFile(stationFile, direction, inboundVirusRate, maxQueueLength, randomSeed)
    trainSchedule = parseTrainScheduleFile(scheduleFile)
    
    logging.basicConfig(filename=logFile, level=logLevel)
    debug("line", stationFile)
    debug("schedule", scheduleFile)
    debug("parameters", iterations, direction, inboundVirusRate, maxQueueLength, randomSeed, plotStats, timeSimulation, logLevel)
    
    output = replicateSimulation(subwayLine, trainSchedule, iterations, direction, plotStats, timeSimulation, antithetic)

    if (outFile != None):
        np.savetxt(outFile, np.asarray(output), delimiter=",")
    
    return output, controlVarResults


if __name__ == "__main__":
    main()
