# R Config
export R_VERSION="4.2.1"
export R_HOME="/usr/local/lib/R"
export DEFAULT_USER="${USERNAME}"

#!/bin/bash
sudo git clone --branch R${R_VERSION} --depth 1 https://github.com/rocker-org/rocker-versioned2.git /tmp/rocker-versioned2
sudo cp -r /tmp/rocker-versioned2/scripts/ /rocker_scripts/
sudo chmod -R 755 /rocker_scripts/
sudo /rocker_scripts/install_R_source.sh

# Use RStudio's package manager to download packages as binaries
export CRAN="https://packagemanager.rstudio.com/cran/__linux__/focal/latest"

# Set up R (RSPM, OpenBLAS, littler, addtional packages)
sudo /rocker_scripts/setup_R.sh
# Correction since script does not work properly
cd /usr/local/bin/
sudo rm r
sudo ln -s /usr/local/lib/R/site-library/littler/bin/r .
# CRAN mirror correction
CRAN=${1:-${CRAN:-"https://cran.r-project.org"}}
ARCH=$(uname -m)
UBUNTU_VERSION=$(lsb_release -sc)
CRAN_SOURCE=${CRAN/"__linux__/$UBUNTU_VERSION/"/""}
if [ "$ARCH" = "aarch64" ]; then
    CRAN=$CRAN_SOURCE
fi
sudo touch ${R_HOME}/etc/Rprofile.site
sudo cat echo "options(repos = c(CRAN = '${CRAN}'), download.file.method = 'libcurl')" >>"${R_HOME}/etc/Rprofile.site"
## Set HTTPUserAgent for RSPM (https://github.com/rocker-org/rocker/issues/400)
sudo cat <<EOF >>"${R_HOME}/etc/Rprofile.site"
# https://docs.rstudio.com/rspm/admin/serving-binaries/#binaries-r-configuration-linux
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])))
EOF

# Re-install system libs that may have been removed by autoremove in rocker scripts
sudo /opt/install-system-libs.sh

# Install devtools to install R packages from GitHub
sudo install2.r --error devtools
sudo install2.r --error readr
sudo install2.r --error ggplot2
sudo install2.r --error dplyr
# Clean
sudo rm -rf /var/lib/apt/lists/*