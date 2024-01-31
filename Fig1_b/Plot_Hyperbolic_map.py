#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:19:46 2024
This code reads the edgelists of the unipartite and bipartite networks as well as partition file and 
plot their hyperbolic maps 
@author: complexity-lab1
"""
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tk
import string
#import community
import os
import sys
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import matplotlib.pylab as pylab
params = {'axes.labelsize': 22,         
         'xtick.labelsize':20,
         'ytick.labelsize':20} 
pylab.rcParams.update(params)


#---------------setting format---------------------
color_seq=['#ff3600','#1f77b4', '#5c97e6','#ff850d','#2ca02c',
           '#a82fe0','#aec7e8','#f7b6d2','#dbdb8d','#f26300',
           '#3b00ed','#a60095','#ff690a','#2e00d9','#0055c9']
color_node=['#8DD1C5','#FFFFB5','#BDB9D8','#F98175','#FCB369',
            '#80AFD0','#B2DD6E','#FBCCE4','#D8D8D8','#BA80BB',
            '#E5E5E5']
color_node2=['#2ca02c','#a82fe0','#006DFF','#FF9200',
             '#4CF7C0','#694172','#F4E040','#0677B4','#D08144','#ED9906',
            '#E5E5E5']

mymarker=['s','o','^','v','d','p','*']  
myline= ['-', '--', '-.', ':'] 
dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)]   
abcd=list(string.ascii_lowercase)      
#---------------setting format---------------------
#----Mathematica colors ----
m_blue='#5E81B5'
m_green='#8FB131'
m_mustard='#E19C24'
m_tile='#EC6235'
l_grey='#CCCCC6'
darker_grey = '#C0C0C0'
# --------------------------


# 1 HYPERBOLIC PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_graph(filename):  
    G1 = nx.Graph()    
    with open(filename) as f:        
        for line in f:
            if not line.lstrip().startswith('#'): 
                  data=line.split() 
                  G1.add_edge(str(data[0]),str(data[1]))
    #G1.remove_edges_from(G1.selfloop_edges())
    G1.remove_edges_from(nx.selfloop_edges(G1))    
    return G1
def read_kappa_theta_r(filename):    
    kappa={}
    theta={}
    r={}
    beta=0
    mu=0    
    with open(filename) as f:        
        for line in f:
            if not line.lstrip().startswith('#'): 
                data=line.split() 
                node=str(data[0])
                kappa[node]=float(data[1])
                theta[node]=float(data[2])
                r[node]=float(data[3])
            else:
                data=line.split() 
                if len(data)==4:
                    if data[2]=="beta:":
                        beta=float(data[3])
                    if data[2]=="mu:":
                        mu=float(data[3])
    return beta, mu, kappa, theta, r
def read_partition(filename):    
    partition={}    
    with open(filename) as f:        
        for line in f:
            if not line.lstrip().startswith('#'): 
                data=line.split() 
                node=str(data[0])
                partition[node]=str(data[1])                
    return partition

def connection_prob(thetai,thetaj,kappai,kappaj,R,beta,mu):
    dtheta=np.pi-abs(np.pi-abs(thetai-thetaj))
    xij=R*dtheta/(mu*kappai*kappaj)
    pij=1.0/(1+xij**beta)
    #if(pij>0.4):
       # print("xij=", xij)
    return pij

def change_coord(G,theta, r):
    x={}
    y={}    
    for i in G.nodes():
        x[i]=r[i]*np.cos(theta[i])
        y[i]=r[i]*np.sin(theta[i])      
        
    return x,y
def set_node_size(r,a,b):  
    r_max=max(r.values())
    node_size={}
    for i in r.keys():
        node_size[i]=a+(b-a)*(1-r[i]/r_max)
    return node_size

def set_node_color_with_partition(name):
    nameset={}
    numb=0
    colorlist={}
    for i in name.keys():
        if name[i] not in nameset.keys():
            nameset[name[i]]=numb
            numb=numb+1
        colorlist[i]=nameset[name[i]]
    return colorlist,nameset
#-------------------------------------------------------------------------------------
def plot_hypermap(file_edgelist,file_coor,file_partition,Tc,Zmin,Zmax, ax1, title, Label, colors):
    
    rx, ry = 3.0, 3.0#3*np.sin(rotation_angle/180.0*np.pi)
    area = rx * ry * np.pi
    theta_angle = np.arange(0, 2 * np.pi + 0.1, 0.1)
    verts = np.column_stack([rx / area * np.cos(theta_angle), ry / area * np.sin(theta_angle)])
     
    G=read_graph(file_edgelist)
    beta, mu, kappa, theta, r=read_kappa_theta_r(file_coor)  
   
    #N_nodes, F_nodes = bipartite.sets(G)
    #N_nodes= len(N_nodes)
    #F_nodes= len(F_nodes)
    N_nodes=2000
    F_nodes=500
    
    
    #N=len(G)    
    R=N_nodes/(2*np.pi)    
    
    x,y=change_coord(G,theta,r)
  
    
    #set node size    
    nodesize=set_node_size(r,Zmin,Zmax)
    #print("*************")
    
    #set node color
    if file_partition=="Null" or file_partition=="null":
        colorlist={}
        for i in G.nodes():
            colorlist[i]=0
        colors1=['#5c97e6','#ff850d','#2ca02c','#ff3600','#1f77b4']
        markers={}
        for i in range(0, N_nodes):
            markers[str(i)] = 'o'  # Default circle markers for nodes in [0, N_Nodes-1]
        for i in range(N_nodes, N_nodes + F_nodes):
            markers[str(i)] = 's'  # Default square markers for nodes in [N_Nodes,  N_Nodes+F_Nodes-1] 
            
        
        
    else:
        #print("WE ARE HERE!!!!!")
        markers={}
        partition=read_partition(file_partition)
        colorlist,nameset=set_node_color_with_partition(partition)
        
        # Determine node colors and markers
        np.random.seed(42)
        colors1=colors
        colors1 = np.vstack([colors1, [0.9, 0.8, 0.95, 1]])
        #print(colors1)
       # colors1=['#5c97e6','#ff850d','#2ca02c','#ff3600','#1f77b4', '#ff69b4', '#e6ccf2']
        for i in range(0, N_nodes):
            markers[str(i)] = 'o'  # Default circle markers for nodes in [0, N_Nodes-1]
            #print( markers[str(i)])
        for i in range(N_nodes, N_nodes + F_nodes):
            colorlist[str(i)]= colors1.shape[0]-1          
            markers[str(i)] = 's'  # Default square markers for nodes in [N_Nodes,  N_Nodes+F_Nodes-1]     
   
    #plot edges

    lines=[]
    G_cut=nx.Graph() 
    ccc=0
    col_line=[]
    for u,v in G.edges(): 
       
        pij=connection_prob(theta[u],theta[v],kappa[u],kappa[v],R,beta,mu)
        
        if pij>Tc: 
            ccc=ccc+1

            lines.append([(x[u], y[u]), (x[v], y[v])])
            if (int(u)<int(v)):
                col_line.append(colors1[colorlist[u]])
            else:
                col_line.append(colors1[colorlist[v]])
            G_cut.add_edge(u,v)
    lc = LineCollection(lines, colors=col_line,linewidths=0.3,linestyles='solid',zorder=1)
    ax1.add_collection(lc)

    
    
    for i in G_cut.nodes():
        #print(i)
        a = [x[i]]
        b = [y[i]]      
        markerrrr = markers[i]  # Get marker shape for the current node
        ax1.scatter(a, b, marker=markerrrr, color=colors1[colorlist[i]], linewidths=0.1, edgecolor='k', s=nodesize[i], zorder=2)



    # Draw a circle 
    r_cut={}
    for i in G_cut.nodes():
        r_cut[i]=r[i]
    rmax=max(r_cut.values())    
    theta_angle=np.linspace(0, np.pi, 100)
    xc=[]
    y_upper = []
    y_lower = []
    for thetax in theta_angle:     
        xc.append(rmax*np.cos(thetax))
        y_upper.append(rmax*np.sin(thetax))
        y_lower.append(-rmax*np.sin(thetax) )
    ax1.fill_between(xc, y_upper, y_lower,linewidth=1.5, facecolor=color_seq[8], 
                     edgecolor='k',zorder=-5, alpha=0.2)

    ax1.set_ylim(-rmax,rmax)
    ax1.set_xlim(-rmax,rmax)
    ax1.axis('off')
    ax1.axis('equal')
    ax1.set_title(title,fontsize=30)
    return ax1

def connection_prob2(thetai,thetaj,kappai,kappaj,R,beta,mu):
    dtheta=np.pi-abs(np.pi-abs(thetai-thetaj))
    xij=R*dtheta/(mu*kappai*kappaj)
    pij=1.0/(1+ (xij**beta))
    return  xij, pij

def plot_hypermap_unipartite(file_edgelist,file_coor,file_partition,Tc,Zmin,Zmax, ax1, title, Label):
    
    rx, ry = 3.0, 3.0#3*np.sin(rotation_angle/180.0*np.pi)
    area = rx * ry * np.pi
    theta_angle = np.arange(0, 2 * np.pi + 0.1, 0.1)
    verts = np.column_stack([rx / area * np.cos(theta_angle), ry / area * np.sin(theta_angle)])
   
    G=read_graph(file_edgelist)
    beta, mu, kappa, theta, r=read_kappa_theta_r(file_coor)   
    
    
    N=len(G)    
    R=N/(2*np.pi)    
    
    x,y=change_coord(G,theta,r)
    
    #set node size    
    nodesize=set_node_size(r,Zmin,Zmax)
    
    #set node color
    if file_partition=="Null" or file_partition=="null":
        colorlist={}
        for i in G.nodes():
            colorlist[i]=0
        colors1=['#5c97e6','#ff850d','#2ca02c','#ff3600','#1f77b4']
        
    else:
        partition=read_partition(file_partition)
        colorlist,nameset=set_node_color_with_partition(partition)

        np.random.seed(42)
        colors1 = plt.cm.jet(np.linspace(0,1,len(set(colorlist.values())) ))  #rainbow 82
        #print(colors1)
        
    
    
    #plot edges
    lines=[]
    col_line=[]
    G_cut=nx.Graph()    
    for u,v in G.edges():    
        pij=connection_prob(theta[u],theta[v],kappa[u],kappa[v],R,beta,mu)
        if pij>Tc:
            lines.append([(x[u], y[u]), (x[v], y[v])])
            G_cut.add_edge(u,v)
            col_line.append(colors1[colorlist[v]])
    lc = LineCollection(lines, colors=darker_grey,linewidths=0.3,linestyles='solid',zorder=1)  #Grey links
   
    ax1.add_collection(lc)

    # plot nodes
    for i in G_cut.nodes():
        a=[x[i]]
        b=[y[i]]        
       
        ax1.scatter(a,b, marker='o',color=colors1[colorlist[i]],linewidths=0.1,edgecolor='k',s=nodesize[i],zorder=2)

    # Draw a circle 
    r_cut={}
    for i in G_cut.nodes():
        r_cut[i]=r[i]
    rmax=max(r_cut.values())    
    theta_angle=np.linspace(0, np.pi, 100)
    xc=[]
    y_upper = []
    y_lower = []
    for thetax in theta_angle:     
        xc.append(rmax*np.cos(thetax))
        y_upper.append(rmax*np.sin(thetax))
        y_lower.append(-rmax*np.sin(thetax) )
    ax1.fill_between(xc, y_upper, y_lower,linewidth=1.5, facecolor=color_seq[8], 
                     edgecolor='k',zorder=-5, alpha=0.2)

    ax1.set_ylim(-rmax,rmax)
    ax1.set_xlim(-rmax,rmax)
    ax1.axis('off')
    ax1.axis('equal')
    ax1.set_title(title,fontsize=30)
    
    return ax1, colors1

def main ():
    
    fig, axx = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axx 
    """
    folder="./output_B_s_3_g_s_3.5_Ns_obs_2000_k_s_30_g_n_3.5_k_n_3_g_f_2.1_N_f_500_B_bi_3_nu_0.0_alpha_neg1_N_l_6_i_0"
    
    Edgelist = f"{folder}/Net_N_2000_g_3.50_B_3.00_k_30.00_a_-1.00_nc_6.unipartite.edgelist"
    Edgelist_Bi = f"{folder}/Net_N_2000_N2_500_g1_3.50_g2_2.10_B_3.00_k1_3.00_nu_0.00.bipartite.edgelist"
    """
    
    folder = [f.path for f in os.scandir('.') if f.is_dir() and 'output' in f.path]
    for index, folder_path in enumerate(folder):

        Edgelist = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(".unipartite.edgelist")]
        Edgelist_Bi = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(".bipartite.edgelist")]
        
    
       
        ax1, colors1=plot_hypermap_unipartite(Edgelist[0], f"{folder[index]}/Unipartite.Coords", f"{folder[index]}/alpha-1.partition", 0.5, 1, 300, ax1,  r'$Unipartite (\alpha={})$'.format(int(-1)), "(a)")
        ax2, colors1 =plot_hypermap_unipartite(Edgelist[0],f"{folder[index]}/Unipartite.Coords", f"{folder[index]}/alpha0.partition", 0.5, 1, 300, ax2, r'$Unipartite (\alpha={})$'.format(int(0)), "(c)")
        ax3, colors1 =plot_hypermap_unipartite(Edgelist[0],f"{folder[index]}/Unipartite.Coords", f"{folder[index]}/alpha5.partition", 0.5, 1, 300, ax3, r'$Unipartite (\alpha={})$'.format(int(5)), "(e)")
       
        ax4=plot_hypermap(Edgelist_Bi[0], f"{folder[index]}/bipartite.Coords" , f"{folder[index]}/alpha-1.partition", 0.5, 1, 300, ax4, r'$Bipartite (\alpha={})$'.format(int(-1)), "(b)", colors1)
        ax5=plot_hypermap(Edgelist_Bi[0], f"{folder[index]}/bipartite.Coords" , f"{folder[index]}/alpha0.partition", 0.5, 1, 300, ax5,  r'$Bipartite (\alpha={})$'.format(int(0)), "(d)", colors1)
        ax6=plot_hypermap(Edgelist_Bi[0], f"{folder[index]}/bipartite.Coords" , f"{folder[index]}/alpha5.partition", 0.5, 1, 300, ax6,  r'$Bipartite (\alpha={})$'.format(int(5)), "(f)", colors1)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.1)
        plt.savefig(f'{folder[index]}/Hyperbolicmap.png',bbox_inches='tight',dpi=300)#fig.dpi 
        plt.show()

if __name__ == "__main__":
    main()
