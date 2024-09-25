import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import statistics
from scipy.stats import binned_statistic

#depth indices that correspond to 15m, 40m, 87.5m, 137.5m, 225m, 350m, 550m, 750m, 950m, 1150m, 1350m, 1625m, and 2250m
depth_list = [2,4,6,8,10,12,14,16,18,20,22,24,26] 

region_dict = { #used for plot titles
            'indian': 'Indian Ocean',
            'southernocean': 'Southern Ocean',
            'northatlantic': 'North Atlantic',
            'tropicalatlantic': 'Tropical Atlantic',
            'southatlantic': 'South Atlantic',
            'northpacific': 'North Pacific',
            'ccs': 'California Current Stream',
            'tropicalpacific': 'Tropical Pacific',
            'southpacific': 'South Pacific',
            'gom': 'Gulf of Mexico',
    }

#Represents one hypothetical float array in a region at a set depth, with a certain number of floats, and given percent of these floats that have ph, o2, and chl sensors
class IndividualExperiment():

    price_dict = { #cost of each BGC sensor
        "oxygen": 7000,
        "nitrate": 24000, #not included in data
        "ph": 10000,
        "chlorophyll": 17000, #biooptics sensor includes chlorophyll, backscatter, downwelling irradiance
        #chlorophyll sensor alone costs ~1/3 of this price, may need to adjust later for more accurate results
    }

    base_cost = 22000 #cost of Core Argo float (float body, temperature and salinity sensors)

    #Initialize an individual experiment (one data point with monte carlo iterations averaged together)
    def __init__(self, region, ph, o2, chl, float_num, depth, updatedFilename):
        self.region, self.ph, self.o2, self.chl, self.float_num, self.depth = region, ph, o2, chl, float_num, depth
        self.filename = updatedFilename #self.filename does not include monte carlo iteration (chops off the last _ and number in original filename)
        self.cost = self.calculate_cost()
        self.unconstrained_variance = self.calculate_unconstrained_variance()
    
    def calculate_cost(self):
        base_expense = self.float_num * self.base_cost
        o2_expense = self.o2 * self.float_num * self.price_dict["oxygen"]
        ph_expense = self.ph * self.float_num * self.price_dict["ph"]
        chl_expense = self.chl * self.float_num * self.price_dict["chlorophyll"]
        total_cost = base_expense + o2_expense + ph_expense + chl_expense
        return total_cost

    def calculate_unconstrained_variance(self):
        variance_sum = 0.0
        numIterations = 0
        for i in range(18): # i ranges from 0-17 (max num of monte carlo iterations)
            filepath = '/Volumes/1TB_SSD/tmp/' + self.filename + '_' + str(i) #update to correct location of data
            if not os.path.exists(filepath): #most experiments have less than 18 monte carlo iterations
                break
            if os.stat(filepath).st_size <= 10: #bug fix for truncated pickle data (check if size is less than 10 bytes)
                continue
            data = load(filepath)
            if len(data) != 2: # bug fix for unused files 
                continue
            variance_sum += data[1].sum() 
            numIterations += 1
        variance_avg = variance_sum/numIterations
        return variance_avg

    #grabs the covariance data of individual experiment and plots it
    def plot_cov_data(self, iterationNum):
        data = load(self.filename + '_' + str(iterationNum))
        plt.plot(data[1]) #data[1] is p_hat_out.diagonal(), while data[0] is indices of array's position
        plt.show()
        plt.close()

#Given a region, depth, and variable, load in and plot all individual experiments (hypothetical float arrays)
class OmniBus():

    #Initialize a set of experiments for a given region, depth, and variable to focus on
    def __init__(self, desired_region, desired_depth, desired_variable):
        directory = '/Volumes/1TB_SSD/tmp' #location of random array data
        self.desired_region = desired_region
        self.desired_depth = desired_depth
        self.desired_variable = desired_variable
        self.ex_list = []
        self.uniqueSet = set() #empty set
        for filename in os.listdir(directory):
            if filename[:3] != 'cm4': #unused files
                continue
            region, ph, o2, chl, float_num, depth = self.parse_filename(filename)
            if region == desired_region and depth == desired_depth:
                updatedFilename = filename[:filename.rfind('_')] #cuts off kk iteration
                tupleData = (region, ph, o2, chl, float_num, depth, updatedFilename)
                self.uniqueSet.add(tupleData) #add tuple to set so that duplicate entries will be ignored (e.g. there will be 5 duplicate entries for an experiment with 5 monte carlo iterations)
        for individualTuple in self.uniqueSet: #average out monte carlo iterations of same experiment
            self.ex_list.append(IndividualExperiment(*individualTuple)) 
                
    def print_ex_list():
        for item in self.ex_list:
            print(item.make_filename())

    @classmethod
    def parse_filename(self, filename):
        basename = os.path.basename(filename)
        file_list = basename.split('_')

        if len(file_list) == 10: #region is two words and we're at shallow depth (with chlorophyll)
            dummy1,region1,region2,dummy2,ph,o2,chl,float_num,depth,kk = file_list
            region = region1 + region2 #join region words together i.e. region1 = north, region2 = atlantic
            #remove variable names (alphabetical chars) from numerical values and convert to correct data types:
            ph = float(ph[2:])
            o2 = float(o2[2:])
            chl = float(chl[3:])
            float_num = int(float_num[3:])
            depth = int(depth)
        elif len(file_list) == 9: #2 cases: either one word region at shallow depth or two word region at deep depth
            depth = int(file_list[-2])
            if depth > 8: #two word region at deep depth (without chlorophyll)
                dummy1,region1,region2,dummy2,ph,o2,float_num,dummy3,kk = file_list #dummy3 used because depth is already parsed
                region = region1 + region2 #join region words together i.e. region1 = north, region2 = atlantic
                #remove variable names (alphabetical chars) from numerical values and convert to correct data types:
                ph = float(ph[2:])
                o2 = float(o2[2:])
                chl = 0 #deep depth
                float_num = int(float_num[3:])
            else: #one word region at shallow depth (with chlorophyll)
                dummy1,region,dummy2,ph,o2,chl,float_num,dummy3,kk = file_list
                #remove variable names (alphabetical chars) from numerical values and convert to correct data types:
                ph = float(ph[2:])
                o2 = float(o2[2:])
                chl = float(chl[3:])
                float_num = int(float_num[3:])
        elif len(file_list) == 8: #region is one word and we're at deep depth (without chlorophyll)
            dummy1,region,dummy2,ph,o2,float_num,depth,kk = file_list
            #remove variable names (alphabetical chars) from numerical values and convert to correct data types:
            ph = float(ph[2:])
            o2 = float(o2[2:])
            chl = 0 #deep depth
            float_num = int(float_num[3:])
            depth = int(depth)
        else:
            raise Exception("Problem with filename parsing, file = " + str(file_list))

        return region, ph, o2, chl, float_num, depth

    def plot(self):
        cost_list = [] #x-axis values
        unconstrained_variance_list = [] #y-axis values
        desired_variable_list = [] #color values
        size_list = [] #size of point values (corresponding to float_num)
        for dummy in self.ex_list:
            cost_list.append(dummy.cost)
            unconstrained_variance_list.append(dummy.unconstrained_variance)
            if self.desired_variable == 'ph':
                desired_variable_list.append(dummy.ph)
            elif self.desired_variable == 'o2':
                desired_variable_list.append(dummy.o2)
            elif self.desired_variable == 'chl':
                desired_variable_list.append(dummy.chl)
            size_list.append(dummy.float_num)
        plt.scatter(cost_list, unconstrained_variance_list, s=size_list, c=desired_variable_list, cmap='viridis')
        plt.title("Cost vs. Unconstrained Variance (" + region_dict[self.desired_region] + ", depth = " + str(self.desired_depth) + ", variable = " + self.desired_variable + ")")
        plt.xlabel("Cost")
        plt.ylabel("Unconstrained Variance")
        plt.colorbar()
        plt.show()
        plt.close()
        #Use following line to save the plot (instead of showing):
        #plt.savefig(self.region + '_' + str(self.desired_depth) + '_' + self.desired_variable + '.png', bbox_inches='tight')

    #Find and return all valuable data regarding the pareto frontier 
    def pareto_engine(self, num_bins):
        cost_list = [] #all x-axis values (not just pareto frontier values)
        unconstrained_variance_list = [] #all y-axis values (not just pareto frontier values)
        desired_variable_list = [] #all color values (not just pareto frontier values)
        size_list = [] #all size values (not just pareto frontier values)
        for dummy in self.ex_list:
            cost_list.append(dummy.cost)
            unconstrained_variance_list.append(dummy.unconstrained_variance)
            if self.desired_variable == 'ph':
                desired_variable_list.append(dummy.ph)
            elif self.desired_variable == 'o2':
                desired_variable_list.append(dummy.o2)
            elif self.desired_variable == 'chl':
                desired_variable_list.append(dummy.chl)
            size_list.append(dummy.float_num)
        #Bin the data based on num_bins parameter:
        min_y_values, bins_edges, bin_numbers = binned_statistic(cost_list, unconstrained_variance_list, 'min', num_bins)
        x_list = [] #x-axis of values along the pareto frontier
        color_list = [] #color values along the pareto frontier
        pareto_size_list = [] #size values along the pareto frontier
        counter = 0 #variable used in deletion of bins that don't contain a value (if user enters a num_bins value that is too high)
        for y_val in min_y_values: #min_y_values are y-axis values along pareto frontier
            if np.isnan(y_val): #if bin doesn't have a value, skip
                min_y_values = np.delete(min_y_values, counter) #delete "not a number" value from array
                continue
            index = unconstrained_variance_list.index(y_val) #find corresponding x, color, and size values
            x_list.append(cost_list[index])
            color_list.append(desired_variable_list[index])
            pareto_size_list.append(size_list[index])
            counter += 1
        return cost_list, unconstrained_variance_list, desired_variable_list, bins_edges, x_list, min_y_values, color_list, pareto_size_list
    
    #Plot the pareto frontier, default frontier contains 40 bins (and therefore 40 points) but can be changed with a different argument
    def plot_individual_pareto(self, num_bins = 40):
        cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list = self.pareto_engine(num_bins)
        #If desired, amplify magnitude of size_list here (to make differences in float_num more apparent between points)
        plt.scatter(cost_list, unconstrained_variance_list, c=desired_variable_list, cmap='Greys') #all data in greyscale to see spread
        plt.scatter(x_list, min_y_values, s=pareto_size_list, c=color_list, cmap='viridis') #pareto data in color
        plt.title("Pareto Frontier (" + region_dict[self.desired_region] + ", depth = " + str(self.desired_depth) + ", variable = " + self.desired_variable + ")")
        plt.xlabel("Cost")
        plt.ylabel("Unconstrained Variance")
        plt.colorbar()
        plt.show()
        plt.close()
        #Use following line to save the plot (instead of showing):
        #plt.savefig(self.region + '_' + str(self.desired_depth) + '_' + self.desired_variable + '.png', bbox_inches='tight')

#Subclasses of OmniBus by region
class OmniBusIndian(OmniBus):
    region = 'indian' #has to match up with region as shown in file names of data (excluding underscore)
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusSO(OmniBus):
    region = 'southernocean'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusNAtlantic(OmniBus):
    region = 'northatlantic'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusTropicalAtlantic(OmniBus):
    region = 'tropicalatlantic'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusSAtlantic(OmniBus):
    region = 'southatlantic'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusNPacific(OmniBus):
    region = 'northpacific'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusCCS(OmniBus):
    region = 'ccs'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusTropicalPacific(OmniBus):
    region = 'tropicalpacific'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusSPacific(OmniBus):
    region = 'southpacific'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

class OmniBusGOM(OmniBus):
    region = 'gom'
    def __init__(self, desired_depth, desired_variable):
        super().__init__(self.region, desired_depth, desired_variable)

#Create plots for every region for every depth level for every variable
def plot_all():
    for bus in [OmniBusIndian, OmniBusSO, OmniBusNAtlantic, OmniBusTropicalAtlantic, OmniBusSAtlantic, OmniBusNPacific, OmniBusCCS, OmniBusTropicalPacific, OmniBusSPacific, OmniBusGOM]:
        for depth in depth_list:
            if depth > 8: #ignore chl above depth index 8
                for variable in ['ph', 'o2']:
                    dummy = bus(depth, variable)
                    dummy.plot()
            else:
                for variable in ['ph', 'o2', 'chl']:
                    dummy = bus(depth, variable)
                    dummy.plot()
            

#One level above OmniBus, used to load in and plot OmniBus objects for every depth level given a variable
class SuperClass():
    
    def __init__(self, OmniBusClass, variable):
        self.OmniBusClass = OmniBusClass
        self.variable = variable
        if self.variable != 'o2' and self.variable != 'ph' and self.variable != 'chl':
            raise Exception("Invalid variable type")
        self.OmniBus_list = []
        for depth in depth_list:
            if variable == 'chl' and depth > 8:
                break
            self.OmniBus_list.append(OmniBusClass(depth, variable))

    #Plot all pareto frontiers for a given region and variable at every depth level
    def plot_all_pareto(self, num_bins = 40):
        plot_values = []
        for bus in self.OmniBus_list:
            cost_list, unconstrained_variance_list, desired_variable_list, bin_edges, x_list, min_y_values, color_list, pareto_size_list = bus.pareto_engine(num_bins)
            plot_values.append(color_list)
        plt.pcolormesh(plot_values)
        plt.title("Pareto Frontier (" + region_dict[self.OmniBusClass.region] + ", " + self.variable + ")")
        plt.xlabel("Binned Costs (Numbered 1 through " + str(num_bins) + ")")
        plt.ylabel("Depth Indices")
        plt.colorbar()
        plt.show()
        plt.close()

def load(file_path):
	"""
	Function loads data saved within the folder

	Parameters
	----------
	file_path: the file path of the data saved within the folder

	Returns
	-------
	unpickled data saved within the folder
	"""

	infile = open(file_path,'rb')
	dummy_file = pickle.load(infile)
	infile.close()
	return dummy_file