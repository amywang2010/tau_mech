@echo off
cd /d "C:\Users\Okapi\Downloads\PED Files\tau_mech"
set OMP_NUM_THREADS=2
.venv\Scripts\python.exe -u scripts\sph_audit.py --steps 8000 --every 200 --eq-steps 4000 --mu-solvent 1.0 --mu-droplet 10.0 --configs no_csf,no_xsph,no_artificial_viscosity,no_shepard,no_immiscibility > logs\audit_isolation.log 2>&1
