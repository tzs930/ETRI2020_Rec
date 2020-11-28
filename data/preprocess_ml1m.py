import gzip
from collections import defaultdict
from datetime import datetime
import os

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0
save_dir = 'processed_data'

# dataset_name = 'Beauty'
rf = open('ml-1m/ratings.dat', 'r')
for l in rf.readlines():
    # 1::595::5::978824268
    userID, movieID, rating, timestamp = l.split("::")
    userID = int(userID)
    movieID = int(movieID)
    rating = float(rating)
    # timestap = timestamp.strip()
    line += 1
    # f.write(" ".join( ) + ' \n')
    # asin = l['asin']
    # rev = l['reviewerID']
    # time = l['unixReviewTime']
    
    countU[userID] += 1
    countP[movieID] += 1

print(line)
line = 0
usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
movieidx_to_mid = dict()

rf.close()
rf = open('ml-1m/ratings.dat', 'r')
# for l in parse('reviews_' + dataset_name + '.json.gz'):
for l in rf.readlines():
    line += 1
    userID, movieID, rating, timestamp = l.split("::")
    userID = int(userID)
    movieID = int(movieID)
    rating = float(rating)
    time = timestamp.strip()

    # asin = l['asin']
    # rev = l['reviewerID']
    # time = l['unixReviewTime']
    if countU[userID] < 5 or countP[movieID] < 5:
        continue

    if userID in usermap:
        userid = usermap[userID]
    else:
        usernum += 1
        userid = usernum
        usermap[userid] = userid
        User[userid] = []

    if movieID in itemmap:
        itemid = itemmap[movieID]

    else:
        itemnum += 1
        itemid = itemnum
        itemmap[movieID] = itemid
        movieidx_to_mid[itemnum] = movieID
        
    User[userid].append([time, itemid])
# sort reviews in User according to time

print(line)
rf.close()

import numpy as np
genome_f = open('tag-genome/tag_relevance.dat')
genome_dict = dict()
max_attr = 1128

for l in genome_f.readlines():
    movie_id, attr_id, rel_score = l.split()
    movie_id = int(movie_id)
    attr_id = int(attr_id)
    rel_score = rel_score.strip()
    rel_score = float(rel_score)
    
    if movie_id in genome_dict:
        genome_dict[movie_id][attr_id] = rel_score
    else:
        genome_dict[movie_id] = np.zeros(max_attr, dtype=float)
        genome_dict[movie_id][attr_id] = rel_score


attr_list = []
print(len(movieidx_to_mid.keys()))

midx_f = open(os.path.join(save_dir, 'movie_idx.txt'), 'w')

attr_list.append(np.zeros(max_attr))  # Insert zero attribute on MovieID=0
for movieidx in movieidx_to_mid:
    mid = movieidx_to_mid[movieidx]
    midx_f.write('%d %d\n' % (movieidx, mid))
        # " ".join([movieidx, ]) + '\n')

    if mid in list(genome_dict.keys()):
        attr_list.append(genome_dict[mid])
        
    else:
        attr_list.append(np.zeros(max_attr))

midx_f.close()

import IPython; IPython.embed()

genome_mat = np.vstack(attr_list)
np.save(os.path.join(save_dir, 'genome_mat.npy'), genome_mat) # [3417, 1128]

# genome_mat = np.loadtxt(f_genome, delimiter='\t')  # (num_movies, num_tags, relevance_score)
# for userid in User.keys():
    # User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)
f = open(os.path.join(save_dir, 'ml-1m.txt'), 'w')
# f = open('Beauty.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()