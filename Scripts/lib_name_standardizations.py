# %% [markdown]
# # Name standardization
# 
# One of the major problems in any data-handling situation is the need to match up databases based on names as indices may either not be available or different databases on different systems use incompatible identifiers. An example for insurance is the matching of company names and the logic for storing these names is not consistent (e.g. company names may change or be misspelt and the extension may be abbreviated and the abbreviations are used, such as "Limited" and "Ltd" or "GmbH" and "Gesellschaft mbH" and "G.m.b.H"). 
# 
# In order to perform matching we will have to rely soft or fuzzy matching but we will need to do some cleaning to help the algorithm further, such as tokenizing strings and removing special characters and punctuations.
# 
# This notebook provides demonstration only and we import a set of artificial data that contains messy information, i.e. each employee ID is associated to a company but we state in actuality the subsidiaries. These names are messy and we want to tidy it up into the two companies:
# 
# - XYZ Specialty
# - ABC Solutions
# 
# See the Section "Apply to external" to see how this performs on an CSV file.
# 
# A lot of the ideas come from the following:
# https://www.analyticsinsight.net/company-names-standardization-using-a-fuzzy-nlp-approach/ 

# %%
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn import cluster
from difflib import SequenceMatcher
import re
from collections import Counter


# %%
def remove_special_chars(string_in: str) -> str:
    '''
    Remove various bits of punctuation from input and returns all in lower case

    Input:
        string_in - str
    
    Output:
        string_out - str
    '''
    chars = [' ', ',', ';', ':', '.', ',', '*', '#', '@', '|', '/', '\\', '-', '_', '?', '%', '!', '^', '(', ')']
    default_char = chars[0]

    tmp_str = string_in
    for c in chars[1:]:
        tmp_str = tmp_str.replace(c, default_char)

    # We also remove multiple spaces by using re.sub
    string_out = re.sub(' +', ' ', tmp_str)
    return string_out.lower()

# %%
def generate_similarity_score_matrix(stringVect):
    '''
    Generate the matrix of Similarity Score for each pair of strings given. The Similarity Score is the harmonic mean of FuzzyWuzzy's token_set_ratio and partial_ratio parameters

    Requires:
        fuzzywuzzy.fuzz
    
    Input:
        stringVect - a vector of strings
    
    Output:
        S - a (n,n)-matrix of Similarity Scores where n = len(stringVect)
    '''
    N=len(stringVect)
    S=np.ones((N,N))*100

    # Generater the entries - we add a tiny number to avoid possible divisions by 0 during ou harmonic mean calcs
    for i in range(N):
        for j in range(N):
            s1 = fuzz.token_set_ratio(stringVect[i], stringVect[j]) + 1e-10
            s2 = fuzz.partial_ratio(stringVect[i], stringVect[j]) + 1e-10
            S[i,j] = 2*s1*s2/(s1+s2)
    return S

# %%
def names_clustering(stringVect):
    '''
    Create clusters of most commonly appearing sub-strings and assign them to items passed in.

    Clustering is done on the similarity matrix, which we will call here on our input

    Requires:
        sklearn.AffinityPropagation
        fuzzywuzzy.fuzz

    Input:
        stringVect - vector of strings
    
    Output:
        dfCluster - a dataframe that contains the original stringVect inputs and their associated cluster
    '''

    # Generate the similarity matrix on input
    S = generate_similarity_score_matrix(stringVect)

    # Fit the Affinity Propagation clustering algorithm on similarity matrix, S
    clusters = cluster.AffinityPropagation(affinity='precomputed', random_state=None).fit_predict(S)

    # Create the output dataframe
    dfCluster = pd.DataFrame(list(zip(stringVect, clusters)), columns=['input_names', 'cluster'])
    return dfCluster

testInput = ['abcd enterprise', 'abcd solutions', 'abcd Europe', 'abcd Asia', 'xyzp America', 'xyzp Portugal', 'xyzp Holdings']
out = names_clustering(testInput)
out

# %%
def get_standard_name(dfClustered, namesCol='input_names'):
    '''
    Names each generated cluster according to the longest common substring in each cluster. Multiple modes will be added if present.
    
    Requires:
        difflib.SequenceMatcher
        collections.Counter
        fuzzywuzzy.fuzz

    Input:
        dfCluster - needs column name "cluster"!
        namesCol - the column with which to search for names 
        
    Output:
        dfNamedCluster - similar to dfCluster but with a suitable name.
    '''
    # Clean up the input_names
    dfClustered[namesCol] = dfClustered[namesCol].apply(remove_special_chars)

    # Initialize empty dictionary - we will fill with the cluster enumeration
    dict_standard_names = {}

    for _cluster in dfClustered['cluster'].unique():
        # Filter in in this cluster, retrive the names column (namesCol)
        names = dfClustered[dfClustered['cluster']==_cluster][namesCol].to_list()

        lsCommonSubstring = [] # Pre-init list that will store common substring

        if len(names) == 1:
            # In this trivial case, we just use the actual name here as our 'common substring'
            dict_standard_names[_cluster] = names[0]
        elif len(names) == 0:
            # This should never need executing but in case there is blanks, then we return string 'unknown'
            dict_standard_names[_cluster] = 'Unknown'
        else:
            # In this cluster, compare pairwise matches and obtain the longest in each pair
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    sequenceMatch = SequenceMatcher(None, names[i], names[j])
                    match = sequenceMatch.find_longest_match(0, len(names[i]), 0, len(names[j]))

                    # Add the common bits as a candidate to the list 'lsCommonSubstring' if non-empty
                    if (match.size > 0):
                        lsCommonSubstring.append(names[i][match.a:match.a + match.size].strip())
            
            n_common = len(lsCommonSubstring)
            counts = Counter(lsCommonSubstring)
            get_mode = dict(counts)
            mode = [k for k, v in get_mode.items() if v == max(list(counts.values()))]
            dict_standard_names[_cluster] = ';'.join(mode)
    
    # Tidy up the dictionary of standard names into a dataframe
    df_standard_names = pd.DataFrame(list(dict_standard_names.items()), columns=['cluster', 'standard_name'])

    # Join with the input data
    df = pd.merge(dfClustered, df_standard_names, on='cluster', how='left')

    # Also add in the confidence level
    df['Confidence'] = df.apply(lambda x: fuzz.token_set_ratio(x['standard_name'], x[namesCol]), axis=1)

    return df