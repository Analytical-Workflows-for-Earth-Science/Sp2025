#################################################################
### Explore the structure of the NOAA BSAI trawl survey data. ###
### Make plots to visualize basic patterns.                    ###
#################################################################
args = commandArgs(trailingOnly=TRUE)
species <- args[1]

# Load libraries
require(dplyr)
require(reshape2)
require(ggplot2)
require(lubridate)
require(PNWColors)

# Load data 
data_presence <- read.csv(paste0("data/", species, "_presence.csv"))
data_abundance <- read.csv(paste0("data/",  species, "_density.csv"))


# plot presence data 
ggplot(data_presence, 
       aes(y = Start.latitude..decimal.degrees.,
           x = Start.longitude..decimal.degrees.,
           color = as.factor(presence)))+
  geom_point(size = 0.75)+theme_classic()+
  scale_color_manual(values = pnw_palette("Lake",n=2),
                     name = paste(species, "present"))+
  facet_wrap(~Survey.year)


ggsave(paste0("results/figures/",  species, "_presenece_map.png"),
       height = 7, width = 9)


# plot abundances 
data_presence <- data_presence %>% 
  filter(presence == 0) %>% 
  select(-presence)
  
data_presence$log_density <- min(data_abundance$log_density)-0.1
data_abundance <- rbind(data_abundance,data_presence)

ggplot(data_abundance, 
       aes(y = Start.latitude..decimal.degrees.,
           x = Start.longitude..decimal.degrees.,
           color = log_density))+
  geom_point(size = 0.75)+theme_classic()+
  viridis::scale_color_viridis()+
  facet_wrap(~Survey.year)


ggsave(paste0("results/figures/",  species, "_abundance_map.png"),
       height = 7, width = 9)



