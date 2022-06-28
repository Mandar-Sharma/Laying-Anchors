import re
import json
import glob
import math
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

def log_squash(num):
    if num > 1:
        return np.log(num) + 1
    elif num < -1:
        return -np.log(-num) -1
    else:
        return num


def find_numerals(data_path):
	#Point data_path to -> WikiText103/rawtext
	files = sorted(glob.glob(data_path + "/*"))

	total_tokens = []
	total_numerals = []

	for file in files:
	    print("\n Working: ", file)
	    with open(file, 'r', encoding='utf-8') as fp:
	        lines = fp.read().split('\n')
	    for line in tqdm(lines):
	        sub_contents = line.split(' ')
	        for token in sub_contents:
	            if token != '':
	                if re.search('[a-zA-Z]', token) == None:
	                    is_num = re.findall(r'\b\d+\b',token.replace(',',''))
	                    if is_num != []:
	                        if is_num[0][0] != '0':
	                            total_numerals.append(is_num[0])
	                else:
	                    total_tokens.append(token.lower())

	#Clean-up the found numeral list
	clean_nums = []
	for num in tqdm(total_numerals):
    	if num[0] != '.':
        	if num[0] == '0':
            	for idx, val in enumerate(num):
                	if val != '0':
                    	break
            	clean_nums.append(num[idx:])
        	else:
            	clean_nums.append(num)


	print("\n Total Tokens: ", len(total_tokens))
	print("\n Total Numerals: ", len(total_numerals))
	print("\n Percentage Numerals: ", round((len(total_numerals)/len(total_tokens))*100,2))

	return total_tokens, clean_nums

def gmm(path, components  = 1000, logarithm = False):
	file = open(path,'rb')
	numerals = pickle.load(file)
	file.close()
	#Limit to 10 billion
	numerals = [float(x) for x in numerals if math.floor(math.log(float(x), 10)) <= 10]
	if logarithm == True:
		numerals = np.array([log_squash(x) for x in numerals]) * 10
		assert np.isnan(numerals.reshape(-1, 1)).any() == False
		assert np.isinf(numerals.reshape(-1, 1)).any() == False
		numerals = numerals.reshape(-1, 1)
	else:
		numerals = np.asarray(numerals)
		assert np.isnan(numerals.reshape(-1, 1)).any() == False
		assert np.isinf(numerals.reshape(-1, 1)).any() == False
		numerals = numerals.reshape(-1, 1)

	print("\n Fitting model for n = {}".format(components))
    gmm = GaussianMixture(n_components=components, init_params='random_from_data', random_state = 42).fit(numerals)
    print("\n Running BIC ")
    bic_val = gmm.bic(numerals_log)
    print("\n BIC: {}".format(bic_val))
    print("\n Running AIC ")
    aic_val = gmm.aic(numerals_log)
    print("\n AIC: {}".format(aic_val))
    
    return gmm.means_, bic_val, aic_val



def main():
	#Define your path to the raw dataset
	dataset_path = "./WikiText103/raw"
	total_tokens, total_numerals = find_numerals()
	#Write numerals to your choice of directory
	file = open("./WikiText103/nums",'wb')
	numerals = pickle.dump(total_numerals, file)
	file.close()

	#Run GMMs on this list of numerals - choose appropriate value of Gaussian components
	gmm_means, _, _ = gmm("./WikiText103/nums", components = 1000, logarithm = False)
	gmm_means_log, _, _ = gmm("./WikiText103/nums", components = 1000, logarithm = True)

	#Save as serialized objects
	file = open("./WikiText103/means",'wb')
	numerals = pickle.dump(gmm_means, file)
	file.close()
	file = open("./WikiText103/log_means",'wb')
	numerals = pickle.dump(gmm_means_log, file)
	file.close()	


if __name__ == "__main__":
	main()