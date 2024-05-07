#!/usr/bin/env python
# coding: utf-8

# In[129]:


import json

f = open('json/networkWithLayout.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
# Closing file
f.close()


# In[130]:


networks = data['networks'][0]


# In[131]:


networks.keys()


# In[132]:


len(networks['site'])


# In[133]:


sites = networks['site']


# In[ ]:





# In[134]:


links = networks['links']


# In[ ]:





# In[135]:


equipment_s0 = sites[0]['equipment']


# In[ ]:





# In[136]:


equipment_s0.keys()


# In[137]:


locations_s0 = equipment_s0['location'][0]


# In[138]:


rack_s0 = locations_s0['rack'][0]


# In[139]:


len(rack_s0)


# In[140]:


rack_s0.keys()


# In[141]:


rack_s0


# In[142]:


len(links)


# In[143]:


links.keys()


# In[144]:


network_links = links['networkLinks']


# In[145]:


site_links = links['siteLinks']


# In[146]:


network_links.keys()


# In[ ]:





# In[147]:


network_links['physical'].keys() #take oms link and fiberspansection and add these to the list as well


# In[148]:


len(site_links)
#Each one has format - (site is site's UID)
#dict_keys(['entityType', 'UID', 'label', 'hierarchicalUID', 'site', 'logical', 'physical'])


# In[149]:


sites = {} #site UID : [nodes in site]
nodes = {} #node UID : [ports in node]


# In[168]:


# for i in range(len(site_links)):
#     #print(i.keys())
#     print(site_links[i]['site'])
#     nodes_in_curr_site = []
#     sites[site_links[i]['site']] = []
#     physical = site_links[i]['physical']
#     logical = site_links[i]['logical']
#     #PHYSICAL
#     for j in physical['L0PhysicalSiteLink']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['L0PhysicalSiteLink'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['L0PhysicalSiteLink'].append(j['dstPort'])
#     for j in physical['L2PhysicalSiteLink']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['L2PhysicalSiteLink'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['L2PhysicalSiteLink'].append(j['dstPort'])    
#     for j in physical['SONETPhysicalSiteLink']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['SONETPhysicalSiteLink'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['SONETPhysicalSiteLink'].append(j['dstPort'])    
#     #LOGICAL
#     for j in logical['L0LogicalSiteLink']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['L0LogicalSiteLink'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['L0LogicalSiteLink'].append(j['dstPort'])
#     for j in logical['L2LogicalSiteLink']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['L2LogicalSiteLink'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['L2LogicalSiteLink'].append(j['dstPort'])    
#     for j in logical['SONETLogicalSiteLink']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['SONETLogicalSiteLink'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['SONETLogicalSiteLink'].append(j['dstPort'])   
#     for j in logical['crossConnect']:
#         sites[site_links[i]['site']].append(j['srcDeviceUID'])
#         sites[site_links[i]['site']].append(j['dstDeviceUID'])
#         if j['srcDeviceUID'] not in nodes.keys():
#             nodes[j['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         if j['dstDeviceUID'] not in nodes.keys():    
#             nodes[j['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
#         nodes[j['srcDeviceUID']]['crossConnect'].append(j['srcPort']) 
#         nodes[j['dstDeviceUID']]['crossConnect'].append(j['dstPort'])       
        


# In[167]:


# sites


# In[150]:


fiberspansection = network_links['physical']['fiberSpanSection']
omsLink = network_links['physical']['omsLink']


# In[151]:


#fiberspansection[0]


# In[152]:


for ele in fiberspansection:
    if ele['srcSite'] not in sites.keys():
        sites[ele['srcSite']] = []
    if ele['dstSite'] not in sites.keys():
        sites[ele['dstSite']] = []        
    if ele['srcDeviceUID'] not in nodes.keys():
        nodes[ele['srcDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
    if ele['dstDeviceUID'] not in nodes.keys():
        nodes[ele['dstDeviceUID']] = {'L0PhysicalSiteLink' : [],'L2PhysicalSiteLink' : [],'SONETPhysicalSiteLink' : [], 'L0LogicalSiteLink' : [], 'L2LogicalSiteLink': [], 'SONETLogicalSiteLink': [], 'crossConnect': [], 'fiberspansection': []}
    sites[ele['srcSite']].append(ele['forward']['srcDeviceUID'])
    sites[ele['srcSite']].append(ele['reverse']['dstDeviceUID'])
    sites[ele['dstSite']].append(ele['forward']['dstDeviceUID'])
    sites[ele['dstSite']].append(ele['reverse']['srcDeviceUID'])
    nodes[ele['srcDeviceUID']]['fiberspansection'].append(ele['forward']['sourcePort'])
    nodes[ele['dstDeviceUID']]['fiberspansection'].append(ele['forward']['destinationPort'])
    nodes[ele['srcDeviceUID']]['fiberspansection'].append(ele['reverse']['sourcePort'])
    nodes[ele['dstDeviceUID']]['fiberspansection'].append(ele['reverse']['destinationPort'])


# In[153]:


for key,val in sites.items():
    sites[key] = list(set(val))


# In[166]:


# sites


# In[ ]:





# In[155]:


all_ports = []


# In[156]:


for key,val in nodes.items():
    for subkey,subval in val.items():
        val[subkey] = list(set(subval))    
        all_ports += list(set(subval))
        all_ports = list(set(all_ports))


# In[157]:


len(nodes)


# In[165]:


# nodes


# In[159]:


all_ports


# In[160]:


len(all_ports)


# In[161]:


rows, cols = (len(all_ports), len(all_ports))
ports_graph = [[0 for i in range(cols)] for j in range(rows)]
#print(ports_graph)


# In[164]:


# for i in range(len(site_links)):
#     physical = site_links[i]['physical']
#     logical = site_links[i]['logical']
#     #PHYSICAL
#     for j in physical['L0PhysicalSiteLink']:
#         src_idx = all_ports.index(j['srcPort'])
#         dst_idx = all_ports.index(j['dstPort'])
#         ports_graph[src_idx][dst_idx] = 1
#     for j in physical['L2PhysicalSiteLink']:
#         src_idx = all_ports.index(j['srcPort']) 
#         dst_idx = all_ports.index(j['dstPort'])
#         ports_graph[src_idx][dst_idx] = 1
#     for j in physical['SONETPhysicalSiteLink']:
#         src_idx = all_ports.index(j['srcPort']) 
#         dst_idx = all_ports.index(j['dstPort'])  
#         ports_graph[src_idx][dst_idx] = 1
#     #LOGICAL
#     for j in logical['L0LogicalSiteLink']:
#         src_idx = all_ports.index(j['srcPort']) 
#         dst_idx = all_ports.index(j['dstPort'])
#         ports_graph[src_idx][dst_idx] = 1
#     for j in logical['L2LogicalSiteLink']:
#         src_idx = all_ports.index(j['srcPort']) 
#         dst_idx = all_ports.index(j['dstPort']) 
#         ports_graph[src_idx][dst_idx] = 1
#     for j in logical['SONETLogicalSiteLink']:
#         src_idx = all_ports.index(j['srcPort']) 
#         dst_idx = all_ports.index(j['dstPort'])   
#         ports_graph[src_idx][dst_idx] = 1
#     for j in logical['crossConnect']:
#         src_idx = all_ports.index(j['srcPort']) 
#         dst_idx = all_ports.index(j['dstPort']) 
#         ports_graph[src_idx][dst_idx] = 1    
        


# In[162]:


for j in fiberspansection:
    src_idx = all_ports.index(j['forward']['sourcePort'])
    dst_idx = all_ports.index(j['forward']['destinationPort'])
    ports_graph[src_idx][dst_idx] = 1
    src_idx = all_ports.index(j['reverse']['sourcePort'])
    dst_idx = all_ports.index(j['reverse']['destinationPort'])
    ports_graph[src_idx][dst_idx] = 1


# In[163]:


conn_count = 1
for i in range(len(all_ports)):
    for j in range(len(all_ports)):
        if ports_graph[i][j] ==1:
            print(conn_count,'. ',all_ports[i],'-----',all_ports[j])
            conn_count +=1


# In[169]:


# sites


# In[170]:


nodes


# In[171]:


all_ports


# In[172]:


ports_graph


# In[281]:


# network_links['physical'].keys() #take oms link and fiberspansection and add these to the list as well


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Site Links has 'N' items where 'N' is number of sites. In each Item,we have the following :

# In[89]:


site_links[0]['physical'].keys()


# In[90]:


site_links[0]['logical'].keys()


# In[ ]:





# In[ ]:





# In[ ]:




