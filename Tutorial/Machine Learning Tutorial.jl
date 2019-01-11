using CSV

heat_capacity_dataframe = CSV.read("./heat_capacity_298K.csv"; header=1)
first(heat_capacity_dataframe, 5)

names(heat_capacity_dataframe)

using Plots
histogram(heat_capacity_dataframe[:,2], nbins = 30)
