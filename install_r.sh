#!/bin/bash
# R Config
export R_VERSION="4.2.1"
export R_HOME="/usr/local/lib/R"
export DEFAULT_USER="${USERNAME}"

# Install R
sudo git clone --branch R${R_VERSION} --depth 1 https://github.com/rocker-org/rocker-versioned2.git /tmp/rocker-versioned2
sudo cp -r /tmp/rocker-versioned2/scripts/ /rocker_scripts/
sudo chmod -R 755 /rocker_scripts/
source /rocker_scripts/install_R_source.sh

# Use RStudio's package manager to download packages as binaries
export CRAN="https://packagemanager.rstudio.com/cran/__linux__/focal/latest"

# Set up R (RSPM, OpenBLAS, littler, addtional packages)
source /rocker_scripts/setup_R.sh

sudo install2.r --error readr
sudo install2.r --error ggplot2
sudo install2.r --error dplyr
# Clean
sudo rm -rf /var/lib/apt/lists/*