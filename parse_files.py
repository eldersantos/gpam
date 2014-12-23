import json
import os
import numpy as np
from igraph import *
from math import *
import pyproj


'''
insert into cidades (cidade_id, name, geom) values (0, 'Brasilia',ST_GeomFromText('POINT(-47.9218204 -15.8266910)',4326));
insert into cidades (cidade_id, name, geom) values (1, 'Belem',ST_GeomFromText('POINT(-48.4901799 -1.4557549)',4326));
insert into cidades (cidade_id, name, geom) values (2, 'Belo Horizonte',ST_GeomFromText('POINT(-43.9344931 -19.9166813)',4326));
insert into cidades (cidade_id, name, geom) values (3, 'Curitiba',ST_GeomFromText('POINT(-49.2671370 -25.4289541)',4326));
insert into cidades (cidade_id, name, geom) values (4, 'Porto Alegre',ST_GeomFromText('POINT(-51.2176584 -30.0346471)',4326));
insert into cidades (cidade_id, name, geom) values (5, 'Recife',ST_GeomFromText('POINT(-34.8828969 -8.0578381)',4326));
insert into cidades (cidade_id, name, geom) values (6, 'Rio de Janeiro',ST_GeomFromText('POINT(-43.2007101 -22.9133954)',4326));
insert into cidades (cidade_id, name, geom) values (7, 'Salvador',ST_GeomFromText('POINT(-38.5023040 -12.9730401)',4326));
insert into cidades (cidade_id, name, geom) values (8, 'Sao Paulo',ST_GeomFromText('POINT(-46.6333094 -23.5505199)',4326));
insert into cidades (cidade_id, name, geom) values (9, 'Campinas',ST_GeomFromText('POINT(-47.0626332 -22.9099384)',4326));
insert into cidades (cidade_id, name, geom) values (10,'Fortaleza',ST_GeomFromText('POINT(-38.5266704 -3.7318616)',4326));
insert into cidades (cidade_id, name, geom) values (11,'Goiania',ST_GeomFromText('POINT(-49.2647943 -16.6868912)',4326));
insert into cidades (cidade_id, name, geom) values (12,'Manaus',ST_GeomFromText('POINT(-60.0217314 -3.1190275)',4326));
insert into cidades (cidade_id, name, geom) values (13,'Sao Luis',ST_GeomFromText('POINT(-44.2829046 -2.5391099)',4326));
'''

base_path = './twitter_data/2turno/day24/'

psdb = ['PSDB', 'Aecio', 'A\\u00e9cio', '45', 'psdb']
pt = ['Dilma', '13', 'PT', 'pt', 'dilma', 'never', 'Never']

file_list = os.listdir(base_path)


g = Graph()
g.add_vertices(14)

g.vs['name'] = [
'Brasilia',
'Belem',
'Belo Horizonte',
'Curitiba',
'Porto Alegre',
'Recife',
'Rio de Janeiro',
'Salvador',
'Sao Paulo',
'Campinas',
'Fortaleza',
'Goiania',
'Manaus',
'Sao Luis'
]

g.vs['lat'] = [
-15.8266910,
-1.4557549,
-19.9166813,
-25.4289541,
-30.0346471,
-8.0578381,
-22.9133954,
-12.9730401,
-23.5505199,
-22.9099384,
-3.7318616,
-16.6868912,
-3.1190275,
-2.5391099
]

g.vs['lng'] = [
-47.9218204,
-48.4901799,
-43.9344931,
-49.2671370,
-51.2176584,
-34.8828969,
-43.2007101,
-38.5023040,
-46.6333094,
-47.0626332,
-38.5266704,
-49.2647943,
-60.0217314,
-44.2829046
]

earthRadius = 6367

pos = range(14)

for l in xrange(0, 14):
	p = pyproj.Proj(proj='utm',ellps='WGS84')
	x, y = p(g.vs[l]['lng'], g.vs[l]['lat'])
	pos[l] = [x,-y]


edges = []
trends = []
names = []

for v in range(0, 14):
	trends.append(v)
	trends[v] = []
	g.vs[v]['colorA'] = 0
	g.vs[v]['colorV'] = 0

for i in range(0, len(file_list) - 1):

	with open(base_path + file_list[i], 'r') as f:
		read_data = f.read()
	f.closed

	ilist = read_data.split('\n')
	#print ilist
	list_size = len(ilist)

	if(list_size == 15):
		for j in  range(0, list_size -1):
			jsonObj = json.loads(ilist[j])
			if(i == 0):
				names.append(jsonObj[0]['locations'][0]['name'])
			#print jsonObj[0]['created_at']
			#print jsonObj[0]['locations'][0]['name']	
			for x in range(0, len(jsonObj[0]['trends'])):
				trends[j].append(jsonObj[0]['trends'][x]['name'])

			for z in  range(0, len(jsonObj[0]['trends'])-1):
				if any(s in jsonObj[0]['trends'][z]['name'] for s in pt):
					g.vs[j]['colorV'] += 1
				elif any(s in jsonObj[0]['trends'][z]['name'] for s in psdb):
					g.vs[j]['colorA'] += 1
			

unique_trends = []

for v in range(0, 14):
	unique_trends.append(set(trends[v]))

intersec = unique_trends[0].intersection(unique_trends[1])
soma = 0

for i in xrange(0, 14):
	if(g.vs[i]['colorA'] > g.vs[i]['colorV']):
		g.vs[i]['color'] = 'blue'
	else:
		g.vs[i]['color'] = 'red'

	for x in xrange(0,14):
		U = unique_trends[i].intersection(unique_trends[x])	
		if(len(U) > 0 and x > i):
			edges.append((i,x))
			g.add_edge(i, x, weights=len(U))
	

for x in xrange(0,len(g.es) - 1):
	soma+=g.es[x]['weights']

print 'size:', soma
#g.add_edges(edges)


for x in xrange(0,14):
	print g.closeness(weights='weights')[x]


plot(g, layout = pos, vertex_label=g.vs['name'], edge_width=g.es['weights'], bbox = (700,800))

'''
adj = g.get_adjacency()

ajdlist = g.get_adjlist()

#print adj

#print g.vs[0].es['weights']
#print g.betweenness(weights='weights')
print 'D', g.diameter(weights='weights')
print 'l', np.sum(g.betweenness(weights='weights')) / 14
print 'degree', g.degree()
print 'madegree', g.maxdegree()
print 'strength', np.sum(g.strength(weights='weights'))/14

print 'closeness', np.sum(g.closeness(weights='weights')) /14

#g.write_gml('./day24.gml')
'''


 	
