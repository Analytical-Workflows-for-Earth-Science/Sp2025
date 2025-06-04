#################################################################
### Load the NOAA Berring sea trawl data sets and related     ###
### Environmental variables.                                  ###
#################################################################

args = commandArgs(trailingOnly=TRUE)
species <- args[1]

# Load libraries
require(dplyr)
require(reshape2)
require(ggplot2)
require(lubridate)

# Load data 
data_species <- read.csv("raw_data/CATCH AND HAUL DATA .csv")%>%
  filter(Taxon.common.name == species)

data_haul<- read.csv("raw_data/HAUL_DATA.csv")



# Get data on snow crab abundance
data_species <- data_species %>% select(Haul.ID,
                                            Number.CPUE..no.km2.)

# Get environmental variables from haul dataset
data_haul <- data_haul %>% select(Haul.ID,
                                  Bottom.temperature..degrees.Celsius.,
                                  Surface.temperature..degrees.Celsius.,
                                  Depth..m., Survey.year,
                                  Start.latitude..decimal.degrees.,
                                  Start.longitude..decimal.degrees.,
                                  Date.and.time )


# add Hauls with no crabs to the abundance dataset
no_crab_Haul.IDs  <- data_haul$Haul.ID[!(data_haul$Haul.ID %in% data_species$Haul.ID )]

data_no_species <- data.frame(Haul.ID=no_crab_Haul.IDs,Number.CPUE..no.km2.=0 )

data_species <- rbind(data_no_species, data_species)

# merge the abundance and haul datasets 
data <- merge(data_haul, data_species, by ="Haul.ID")

# convert date and time to day of year
data$Day.of.year <- yday(dmy(substr(data$Date.and.time,1,11)))
data <- data %>% select(-Date.and.time)

#########################################
#### merge in annual variables       ####
#### data sets from Litzo et al 2024 ####
#########################################
bloom_timing <- read.csv("raw_data/bloom_timing.csv")

# Select bloom timing in northern and southern BSAI regions
bloom_timing <- bloom_timing %>% mutate(Survey.year = year) %>%
  select(Survey.year, north_south, globcolour_peak_mean) %>%
  dcast(Survey.year~north_south)

# Rename variables
names(bloom_timing)  <- c("Survey.year","blooms_north","blooms_south")

# Scale values, mean zero standard deviation of one
bloom_timing$blooms_north <- scale(bloom_timing$blooms_north)
bloom_timing$blooms_south <- scale(bloom_timing$blooms_south)

# Bloom type
bloom_type <- read.csv("raw_data/bloom_type.csv")

bloom_type <- bloom_type %>% 
  mutate(Survey.year = year) %>%
  select(-year,-X)%>%
  dcast(Survey.year~north_south+gl_type, value.var = "count")

# Impute NAs to zeros
bloom_type[is.na(bloom_type)] <-0 

# Scale values mean zero and variance one
bloom_type$north_ice_free <- scale(bloom_type$north_ice_free)
bloom_type$north_ice_full <- scale(bloom_type$north_ice_full)
bloom_type$south_ice_free <- scale(bloom_type$south_ice_free)
bloom_type$south_ice_full <- scale(bloom_type$south_ice_full)

# Sea ice extent
sea_ice <- read.csv("raw_data/ice.csv")

# Scale values
sea_ice$JanFeb_ice <- scale(sea_ice$JanFeb_ice)
sea_ice$JMarApr_ice <- scale(sea_ice$MarApr_ice)

# Rename year column 
sea_ice$Survey.year <- sea_ice$year
sea_ice <- sea_ice %>% select(-year)


#################################
### Merge in annual variables ###
#################################

data <- merge(data,bloom_timing,by = "Survey.year")%>%
  merge(bloom_type,by = "Survey.year")%>%
  merge(sea_ice,by = "Survey.year")


# Scale all variables (annual variables are already scaled)
data$Bottom.temperature..degrees.Celsius. <- scale(data$Bottom.temperature..degrees.Celsius.)
data$Surface.temperature..degrees.Celsius. <- scale(data$Surface.temperature..degrees.Celsius.)
data$Depth..m. <- scale(data$Depth..m.)
data$Start.latitude..decimal.degrees. <- scale(data$Start.latitude..decimal.degrees.)
data$Start.longitude..decimal.degrees. <- scale(data$Start.longitude..decimal.degrees.)
data$Day.of.year <- scale(data$Day.of.year)



# Variable for presence
data$presence <- data$Number.CPUE..no.km2. > 0


# variable for abundance
data_numbers <- data %>% filter(presence > 0) %>%
  mutate(log_CPUE = log(Number.CPUE..no.km2.))

data_numbers$log_density <- scale(data_numbers$log_CPUE)

write.csv(data %>% select(-Number.CPUE..no.km2.),
          paste0("data/", species, "_presence.csv"))

write.csv(data_numbers %>% select(-Number.CPUE..no.km2.,-presence, -log_CPUE),
          paste0("data/", species, "_density.csv"))






