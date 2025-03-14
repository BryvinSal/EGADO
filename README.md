# EGADO
Codes of the custom edge-guided analog-and-digital optimization algorithm for the inverse design of silicon photonic devices.
These are supporting materials for "Edge-guided inverse design of digital metamaterial-based mode multiplexers for high-capacity multi-dimensional optical interconnect" published on Nature Communications.  
## Citation
Sun, A., Xing, S., Deng, X. et al. Edge-guided inverse design of digital metamaterial-based mode multiplexers for high-capacity multi-dimensional optical interconnect. Nat Commun 16, 2372 (2025). https://doi.org/10.1038/s41467-025-57689-7.  Any requests and questions can contact Dr.Sun (alsun22@m.fudan.edu.cn)
## Overview
This repository contains the necessary scripts and functions for running the optimization and analysis of photonic devices using Lumerical FDTD simulations. 
## Requirements
The project requires Python 3.6 and specific dependencies outlined below, as well as the Lumerical FDTD software with a valid license. The version used in this project is 2023R2.2.
### Required Python:
Python Version: 3.6
### Required Libraries:
lumopt  
lumapi  
splayout
### Required Additional Software:
Lumerical FDTD: Download and install the Lumerical FDTD software (version 2023R2.2) and ensure access to a valid license.
## File Structure
The repository is structured as follows:

main.py: The main program that orchestrates the optimization process and calls the required functions and scripts. Start your execution here.  
Files starting with func_: These are Python functions that provide modularized and reusable logic for the optimization process.  
Files starting with base_: These are Lumerical script files used for interaction with the Lumerical FDTD software.  
Files starting with appendix_: Auxiliary programs used after the optimization process, such as visualization (e.g., plotting) or additional post-processing.  
## Instructions
Install Python 3.6 and ensure that the required dependencies are installed.  
Download and install the Lumerical FDTD software (2023R2.2). The estimated installing time is 1 hour. Ensure that you have access to a valid license.  
The lumopt and lumapi libraries are installed automatically with the software. Instructions for implementation can be found at https://optics.ansys.com/hc/en-us/articles/360041873053-Session-management-Python-API

## Notes
Ensure that Lumerical's FDTD software is properly installed and added to your system's PATH.  
Make sure your license for Lumerical FDTD is active and valid before running the scripts.  
For troubleshooting, please check the documentation of the libraries (lumopt, lumapi, splayout) and the Lumerical FDTD software.

## Demo
We upload a demo file "demo.fsp" which can be used to reproduce the device performance of the inverse-designed five-mode MUX. The estimated running time is less than 60 seconds using GPU acceleration. To run the .fsp file, Lumerical FDTD must be successfully downloaded with a valid license.
