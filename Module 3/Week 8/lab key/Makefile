# Makefile
# Date: January 29th, 2025
#
DIRPATH=~/documents/
#
#
SPECIES = "Arctic cod"
#
#
NTREE = 100
#
#
MTRY = 3
#
#
VARS = Depth..m. Bottom.temperature..degrees.Celsius. Surface.temperature..degrees.Celsius.
#
#
all:
	@echo ""
	@echo "    This folder is use to run species distribution models on the NOAA berring sea groundfish trawl data set"
	@echo ""
	@echo ""
	@echo "    Please type.....   "
	@echo "    		make run_analysis SPECIES='\"Arctic cod\"' "
#
#
run_analysis:
	Rscript src/process_ground_fish_trawl_data.R $(SPECIES);
	Rscript src/EDA_plots.R $(SPECIES);
	Rscript src/build_random_forest.R $(SPECIES) $(NTREE) $(MTRY);
	Rscript src/varImpPlot.R $(SPECIES) presence;
	Rscript src/varImpPlot.R $(SPECIES) density;
	@for var in $(VARS); do \
	Rscript src/partial_dependence_plot.R $(SPECIES) $${var}; \
	done;