#******************************************************************************#
# Python3.5
# House: /usr/bin/python3.5
# ICM: /usr/local/bin/python3.5
# Mac: /opt/intel/intelpython3/bin/python3.6

#******************************************************************************#
import os   #@UnusedImport
import numpy as np   #@UnusedImport

#******************************************************************************#
f1 = open("./results_fix.txt", 'w')
f1.write('id,is_iceberg\n')

#******************************************************************************#
# 
file_name = "./results.txt"
with open(file_name) as f:
	lines_in_file = f.readlines()

check_header_line = False   	# To avoid read the header line.
for line in lines_in_file:
	if check_header_line:
		
		data = line.rstrip('\n').split(',')
		id_imag = data[0]
		prob = float(data[1])
		
		prob = min(prob,0.9999)
		prob = max(prob,0.0001)
		
		f1.write(id_imag+","+"{:.6f}".format(float(np.squeeze(prob)))+"\n")
		
	else:
		check_header_line = True

f1.close()

#******************************************************************************#
print('End!!!')













