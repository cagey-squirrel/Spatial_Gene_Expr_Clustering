import pandas as pd
import hdbscan
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from hdbscan import HDBSCAN

from sklearn.neighbors import BallTree
from umap import UMAP
import pickle
from time import time
from scipy.spatial import KDTree
from collections import defaultdict

def read_expression_data(data_path):
    '''
    Reads expression data from data_path tsv file into pandas.datafrane

    Inputs:
        - data_path (string) : path to .tsv file 
    
    Returns:
        - df (pandas.DataFrame) : dataframe of read data
    '''
    df = pd.read_csv(data_path, sep = '\t')

    return df


def get_cell_average_coordinates(df):
    '''
    Makes a dataframe which holds cell position for each cell
    Each cell can have multiple coordinates, this function stores the mean of coordinates for each cell

    Input:
        -df (pd.DataFrame) with columns ('cell', 'x', 'y', 'geneID', 'PIDCounts')
    
    Output
        - coordinates_df (pd.DataFrame) with columns ('cell', 'x', 'y')
    '''

    coordinates_df = df.groupby('cell')[['x', 'y']].mean()
    return coordinates_df


def pivot_df_table(df):
    '''
    Pivots the df table so each gene has its own column

    Input:
        - df (pd.DataFrame) size = (count_of_present_genes) * (4_column)

    Returns:
        - df (pd.DataFrame) size = (num_of_cells) * (num_of_genes)
    '''

    pivoted_df = df.pivot_table('MIDCounts', 'cell', 'geneID').fillna(0)
    return pivoted_df


def check_unique_genes(df):
    '''
    This function was used to check whether same gene can appear in the same cell on multiple positions
    Anser is no
    '''
    sum_unique_genes = 0
    sum_non_unique_genes = 0

    for id in df['cell'].unique():

        new_df = df[df['cell'] == id]
        len_df = len(new_df)
        len_uniq = len(new_df['geneID'].unique())

        if len_df == len_uniq:
            sum_unique_genes += 1
        else:
            sum_non_unique_genes += 1
    
    print(f'sum1 = {sum_unique_genes}, sum2 = {sum_non_unique_genes}')


def sum_cell_gene_counts(df):
    '''
    Sums gene expressions in a single cell, grouped by geneID
    Each cell now only appears once in a table
    X and Y values are averages of all X and Y values from cell 
    
    Input:
        - df (pd.DataFrame)
    
    Returns:
        - sum_df (pd.DataFrame)
    '''
    
    grouped__by_cell_and_gene_df = df.groupby(['cell', 'geneID'], as_index=False)

    sum_df = grouped__by_cell_and_gene_df['MIDCounts'].sum() # MIDCounts are sums grouped by both cell and gene ids
    # X and Y are averages of all rows with given cell id
    for cell_id in df['cell'].unique():
        cell_df = df[df['cell'] == cell_id][['x', 'y']]
        x_average = cell_df['x'].mean()
        y_average = cell_df['y'].mean()

        sum_df.loc[sum_df['cell'] == cell_id, 'x'] = x_average
        sum_df.loc[sum_df['cell'] == cell_id, 'y'] = y_average

    return sum_df


def segmentation_cmap(seed):
    '''
    Making colormap with random colors for clusters
    '''
    np.random.seed(seed)
    vals = np.linspace(0, 1, 50)
    np.random.shuffle(vals)
    return plt.cm.colors.ListedColormap(plt.cm.CMRmap(vals))


def get_new_cluster_label(coordinates_df, cell_row, indices):
    '''
    This function calculates new label for a cell based on its nearest neighbors

    Vector cluster_metrics holds values which indicate how much our cell is similar to each cluster
    In the end argmax of this vector is chosen as the new cluster label for the cell

    For each neighbor we check neighbors cluster_label and calculate similarity between our cell and that neighbor
    We add this similarity to the cluster_metrics[neighbors_cluster_id]

    Input:
        - coordinates_df (pd.DataFrame) : data which holds information about cells position, gene expression and cluster id
        - cell_row (row from pd.Dataframe) : has columns ['CMP1', 'CMP2'] for 2 UMAP components
        - indices (array-like) : stores indices of k-nearest-neighbors

    Returns:
        - cluster_label (int) : calculated new cluster label

    '''
    cluster_metrics = defaultdict(lambda: 0)
    

    for index in indices[1:]:  #  We skip first index because cells closest neighbor is the cell itself
        
        neighbor = coordinates_df.iloc[index]
        neighbor_cluster_label = neighbor['cluster_labels']
        expression_similarity = 2 - get_expression_difference(cell_row, neighbor)  # similarity is opposite of expression difference
        cluster_metrics[int(neighbor_cluster_label)] += expression_similarity
    
    cluster_label = max(cluster_metrics, key=cluster_metrics.get)
            
    return cluster_label
    

def get_expression_difference(cell, neighbor_cell):
    '''
    Calculates the differene in gene expression between cell and neighbor cell
    Difference is calculated as Euclidean distance between cells UMAP components
    Input:
        - cell (row from pd.Dataframe) : has columns ['CMP1', 'CMP2'] for 2 UMAP components
        - neighbor (row from pd.Dataframe) : has columns ['CMP1', 'CMP2'] for 2 UMAP components
    
    Returns:
        - expression_difference (float) : difference in expression between two cells
    '''
    return ((cell['PCA1'] - neighbor_cell['PCA1'])**2 + (cell['PCA2'] - neighbor_cell['PCA2'])**2)**0.5


def spatial_clustering(coordinates_df, k=5, num_iterations=float('inf')):
    '''
    This function fine-tunes output from clustering
    For each cell it looks at 5 nearest neighborts from given cell and decides which cluster this cell should belong to
    If there is sufficient evidence that most of the neighboring cells belong to different cluster then cluster_id is changed
    This process is repeated for every cell as long as there are changes in cluster labels or for num_iteration iterations
    Input:
        - coordinates_df (pd.DataFrame) : data which holds information about cells position, gene expression and cluster id
        - k (int) : number of nearest neighbors to visit
        - num_iterations (int) : number of iteration for which the algorithm runs. Default is run until convergence
    Return:
        - coordinates_df (pd.DataFrame) : data with refined cluster labels
    '''
    
    i = 0
    
    # Coordinates normalization -> not needed unless distance is used in calculating new cluster label
    coordinates_df['x'] = (coordinates_df['x'] - coordinates_df['x'].min()) / (coordinates_df['x'].max() - coordinates_df['x'].min())
    coordinates_df['y'] = (coordinates_df['y'] - coordinates_df['y'].min()) / (coordinates_df['y'].max() - coordinates_df['y'].min())
    
    # KDTree used for finding nearest neighbors
    tree = KDTree(coordinates_df[['x', 'y']])

    while(i < num_iterations):
        new_labels = coordinates_df['cluster_labels'].copy()

        for cell_index, cell_row in coordinates_df.iterrows():
            
            distances, indices = tree.query(cell_row[['x', 'y']], k)
            cluster_label = get_new_cluster_label(coordinates_df, cell_row, indices)
            new_labels.loc[cell_index] = cluster_label
        
        # If there werent any changes algorithm has converged
        if (new_labels == coordinates_df['cluster_labels']).all():
            break
        else:
            print(f'Number of changed labels = {sum(new_labels != coordinates_df["cluster_labels"])}')

        i += 1
        coordinates_df['cluster_labels'] = new_labels
    
    return coordinates_df


def main():
   
    # Loading data
    #--------------------------------------------------------------------
    print('Started loading...')
    data_path = 'dorsal.tsv'
    df = read_expression_data(data_path)
    coordinates_df = get_cell_average_coordinates(df)
    original_coordinates_df = coordinates_df.copy()
    df = pivot_df_table(df)
    print(f'Finished loading: shape is {df.shape} \n')
    #--------------------------------------------------------------------




    # Plotting QC features
    #--------------------------------------------------------------------
    print(f'Started feature plotting')
    cells_per_gene_df = df.astype(bool).sum(axis=0)
    cells_per_gene_df.plot.hist(title='Num of cells per gene')
    plt.savefig('cells_per_gene.png')
    plt.clf()

    # How much counts does each cell have
    counts_per_cell_df = df.sum(axis=1)
    counts_per_cell_df.plot.hist(title='Num of counts per cell')
    plt.savefig('counts_per_cell.png')
    plt.clf()
    print(f'Finished feature plotting: images saved at images folder \n')
    #--------------------------------------------------------------------




    # Filtering
    #--------------------------------------------------------------------
    print('Started filtering...')
    old_df_shape = df.shape
    filter_vector = (counts_per_cell_df > 200) & (counts_per_cell_df < 2000)
    df = df[filter_vector]
    coordinates_df = coordinates_df[filter_vector]
    print(f'filtered {(1 - (filter_vector.sum() / len(filter_vector))) * 100}% of cells')

    filter_vector = cells_per_gene_df > 10
    df = df.loc[:, filter_vector]
    print(f'filtered {(1 - (filter_vector.sum() / len(filter_vector))) * 100}% of genes')

    print(f'Finished filtering -> Old shape = {old_df_shape} and new shape = {df.shape} \n')
    #--------------------------------------------------------------------



    # Normalization
    #--------------------------------------------------------------------
    print(f'Started normalization...')
    df = df.div(df.sum(axis=1), axis=0)
    print(f'Finished normalization \n')
    #--------------------------------------------------------------------

    file = open('original_coordinates_df.pickle', 'wb')
    pickle.dump(coordinates_df, file)
    



    # Dimesionality reduction
    #--------------------------------------------------------------------
    print('Started dimesionality reduction')
    #pca = PCA(n_components=10)
    #df_components = pca.fit_transform(df)

    umap_cls = UMAP(n_neighbors=5, min_dist=0.0, n_components=2, random_state=1303)
    df_components = umap_cls.fit_transform(df)

    coordinates_df['PCA1'] = df_components[:, 0]
    coordinates_df['PCA2'] = df_components[:, 1]

    plt.scatter(coordinates_df['PCA1'], coordinates_df['PCA2'], s=1)
    plt.savefig('just_components.png')
    plt.clf()

    print('Finished dimesionality reduction: dimensions reduced to 2. Image saved at images folder \n')
    #--------------------------------------------------------------------
 



    # Clustering
    #--------------------------------------------------------------------
    print('Started clustering...')
    hdb = DBSCAN(eps=0.15, min_samples=10)
    hdb.fit(coordinates_df[['PCA1', 'PCA2']])
    print(f'Finished clustering: number of clusters is {len(set(hdb.labels_))}\n')

    print('Started filtering clusters')
    for cluster_label in list(set(hdb.labels_)):
        label_points = coordinates_df[hdb.labels_ == cluster_label]
        if len(label_points) < 25:
            hdb.labels_[hdb.labels_ == cluster_label] = -1

        print(f'Cluster of label {cluster_label} is the size of {len(label_points)}')

    coordinates_df = coordinates_df[hdb.labels_ != -1]
    hdb.labels_ = hdb.labels_[hdb.labels_ != -1]
    coordinates_df['cluster_labels'] = hdb.labels_
    #original_coordinates_df['labels'] = -1
    print(f'Finished filtering clusters: new number of clusters is {len(set(hdb.labels_))}')

    fig = plt.figure(figsize=(15, 10))
    plt.scatter(coordinates_df['y'], coordinates_df['x'], c=coordinates_df['cluster_labels'], s=10, cmap=segmentation_cmap(2))
    plt.savefig('clustered_cells_before1.png')
    plt.clf()
    
    # Fine-tuning
    coordinates_df = spatial_clustering(coordinates_df, 5, 1)

    print('Started plotting clusterization...')
    
    fig = plt.figure(figsize=(15, 10))
    plt.scatter(coordinates_df['y'], coordinates_df['x'], c=coordinates_df['cluster_labels'], s=10, cmap=segmentation_cmap(2))
    plt.savefig('clustered_cells1.png')
    plt.clf()
    print('Finished plotting clusterization')
    #--------------------------------------------------------------------


if __name__ == '__main__':
    main()



