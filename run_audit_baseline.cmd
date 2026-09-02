@echo off
cd /d "C:\Users\Okapi\Downloads\PED Files\tau_mech"
set OMP_NUM_THREADS=2
.venv\Scripts\python.exe -u scripts\sph_audit.py --steps 50765 --every 500 --eq-steps 4000 --mu-solvent 1.0 --mu-droplet 10.0 --configs baseline > logs\audit_baseline_full.log 2>&1
