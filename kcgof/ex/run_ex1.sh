#!/bin/bash 

screen -AdmS e1kcgof -t tab0 bash 

# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

# screen -S e1kcgof -X screen -p 1 bash -lic "conda activate kcgof; python ex1_vary_n.py quad_quad_d1"
# screen -S e1kcgof -X screen -p 2 bash -lic "conda activate kcgof; python ex1_vary_n.py quad_vs_lin_d1"
# screen -S e1kcgof -X screen -p 2 bash -lic "conda activate kcgof; python ex1_vary_n.py g_het_dx5"
# screen -S e1kcgof -X screen -p 2 bash -lic "conda activate kcgof; python ex1_vary_n.py gaussls_h0_d5"
# # screen -S e1kcgof -X screen -p 2 bash -lic "conda activate kcgof; python ex1_vary_n.py gauss_t_d1"

python ex1_vary_n.py g_het_dx3
python ex1_vary_n.py gaussls_h1_d1_easy
python ex1_vary_n.py gaussls_h0_d1
python ex1_vary_n.py gaussls_h0_d5
python ex1_vary_n.py quad_vs_lin_d1

# python ex1_vary_n.py gaussls_h0_d5

# python ex1_vary_n.py quad_quad_d1
# python ex1_vary_n.py quad_vs_lin_d1
# python ex1_vary_n.py gauss_t_d1

