import sklearn
import sklearn.cluster 
import sklearn.mixture 
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.linalg import inv
import matplotlib as mpl
# mpl.style.use('/app/ai_experiments/Utils/sp_custom.mplstyle')

def cluster_img(data, true_lbl, method: str, test_data = None, test_lbl = None, binary_th: float=False, visualize = True, **kwargs):
    
    if method == 'kmeans':
        clustering_model = sklearn.cluster.KMeans(**kwargs).fit(data)

    elif method == 'dbscan':
        clustering_model = sklearn.cluster.DBSCAN(**kwargs).fit(data)

    elif method == 'gm':
        clustering_model = sklearn.mixture.GaussianMixture(**kwargs).fit(data)

    
    if hasattr(clustering_model, "labels_"):
        y_pred = clustering_model.labels_.astype(int)
    else:
        y_pred = clustering_model.predict(data)

    if any(y_pred == -1):
        print(f"{method} didn't converge")
        return
    else:
        print(f"{method} fitted. Evaluating train data...\n")
        evaluate_cluster(data, y_pred, true_lbl)

    if visualize == True:
            visualize_cluster_dist(y_pred, true_lbl, plot_t='kde')

    if test_data is not None:
        out_of_sample_pred = clustering_model.predict(test_data)
        evaluate_cluster(test_data, out_of_sample_pred, test_lbl, binary_th)

        if visualize == True:
            visualize_cluster_dist(out_of_sample_pred, test_lbl, plot_t='kde')

        return clustering_model, out_of_sample_pred

    return clustering_model

def display_images_cluster(c, th, data, pred_lbl, true_lbl, num_images, name):

    dir_path = f"/app/ai_experiments/Lockout/tmp"
    num_rows = np.ceil(num_images / 4).astype(int)
    fig, axs = plt.subplots(num_rows, 4)
    try:
        for i in range(num_images):
            cluster_data = data[(pred_lbl == c) & (true_lbl >= th[0]) & (true_lbl <= th[1])]
            img = cluster_data[i][0].squeeze()
            if num_rows > 1:
                axs[i//4, i%4].imshow(img)
                axs[i//4, i%4].set_title(f'Label: {true_lbl[(pred_lbl==c) & (true_lbl >= th[0]) & (true_lbl <= th[1])][i]}')
            else:
                axs[i].imshow(img)
                axs[i].set_title(f'Label: {true_lbl[(pred_lbl==c) & (true_lbl >= th[0]) & (true_lbl <= th[1])][i]}')

        img_path = f"{dir_path}/{name}_cluster_fig_{c}.png"

        plt.savefig(img_path)  
        plt.show()
        plt.close(fig)
        
    except:
        print('error')   

def visualize_cluster(data, pred_lbl, true_lbl, num_images = 8, name = '', display_all=False):

    clusters = set(pred_lbl)
    print(f'Total clusters: {len(clusters)}')
    for c in clusters:
        print(f'cluster: {c}/{len(clusters)}')

        if display_all:
            ths = [[0, 0.5], [0.5, 1.0]]
            for th in ths:
                display_images_cluster(c, th, data, pred_lbl, true_lbl, num_images, name)
        else:
            th = [0.0, 1.0]
            display_images_cluster(c, th, data, pred_lbl, true_lbl, num_images, name)

def visualize_cluster_dist(pred_lbl, true_lbl, plot_t='kde', name = ''):
    dir_path = f"/app/ai_experiments/Lockout/tmp"
    clusters = set(pred_lbl)

    for c in clusters:
        data_cluster = true_lbl[pred_lbl == c]
        if plot_t == 'kde':
            sns.kdeplot(data=data_cluster, label=f'Cluster {c}')
        elif plot_t == 'hist':
            sns.histplot(data=data_cluster, kde=True, label=f'Cluster {c}', alpha=0.7)
        
    plt.xlabel('ModIOU Values')

    if plot_t == 'kde':
        plt.xlim([0,1])
        plt.ylabel('Density')
        plt.title('Kernel Density Estimate (KDE) of ModIOU Values for Different Clusters')

    elif plot_t == 'hist':
        plt.ylabel('Frequency')
        plt.title('Histograms of ModIOU Values for Different Clusters')

    plt.legend()

    img_path = f"{dir_path}/{name}_cluster_dist_fig.png"
    plt.savefig(img_path)  
    plt.show()

def clusters_acc(cm, r):
    TP = cm[r][0]
    FP = cm[r][1]
    TN = cm[1-r][1]
    FN = cm[1-r][0]
    precision = TP/(TP+FP)
    sensitivity = TP/(TP+FN)
    return (precision, sensitivity)

def match_clusters(cm):
    p1, s1 = clusters_acc(cm, 0)
    p2, s2 = clusters_acc(cm, 1)
    p = [p1, p2]
    s = [s1, s2]
    precision = max(p)
    sensitivity = s[np.argmax(p)]
    return (precision, sensitivity)

def evaluate_cluster(X, pred_labels, true_labels, binary_th: float=False):
    
    if binary_th != False:
        true_labels = np.where(true_labels > binary_th, 1.0, 0.0)
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        (precision, sensitivity) = match_clusters(cm) #combination that maximizes TN
        print(f"Precision: {precision}; Sensitivity: {sensitivity}")
    
    silhouette_score = metrics.silhouette_score(X, pred_labels)
    ari_score = metrics.adjusted_rand_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
    calinski_harabasz_score = metrics.calinski_harabasz_score(X, pred_labels)
    
    print("Silhouette Score:", silhouette_score)
    print("ARI Score:", ari_score)
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
    print("Calinski-Harabasz Score:", calinski_harabasz_score)

def mahalanobis_distance(data, new_samples):
    mean_vec = np.mean(data, axis=0)
    cov_mat = np.cov(data, rowvar=False)
    delta = 1e-6 # this value can be changed according to your need
    cov_mat_reg = cov_mat + np.eye(cov_mat.shape[0]) * delta
    cov_mat_inv = inv(cov_mat_reg)

    delta_x = new_samples - mean_vec #vector for each new sample
    return np.sqrt(np.dot(np.dot(delta_x, cov_mat_inv), delta_x.T))

def cluster_info(cluster_pred, cluster_gt_binary, cluster_gt, v = False, show_img = None):
    """
    Get information about the cluster: percentage of points belonging to the dominant class and the dominant class.
    
    Parameters:
        - cluster_labels: List of labels in the cluster.
        - cluster_pred: predicted cluster class

    Returns:
        - max_class_percentage: Percentage of points in the cluster that belong to the dominant class.
        - dominant_class: The class with the highest number of points in the cluster.
    """
    clusters = set(cluster_pred)
    total = len(cluster_pred)

    if v:
        print(f'Total samples: {total}')

    stats = {}
    for c in clusters:
        cluster_labels = cluster_gt_binary[cluster_pred == c]
        cluster_labels_raw = cluster_gt[cluster_pred == c]
        total_points = len(cluster_labels)
        if total_points == 0:  # Empty cluster
            return 0, None
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        dominant_index = np.argmax(counts)
        max_class_percentage = (counts[dominant_index] / total_points) * 100
        dominant_class = unique_labels[dominant_index]
        
        probabilities = counts / total_points
        gini = 1 - sum([p**2 for p in probabilities])

        min_IOU = min(cluster_labels_raw)
        max_IOU = max(cluster_labels_raw)
        mean_IOU = np.round(np.mean(cluster_labels_raw), 3)
        Q1 = np.percentile(cluster_labels_raw, 25)
        Q3 = np.percentile(cluster_labels_raw, 75)
        IQR_IOU = np.round(Q3 - Q1, 3)

        stats[f'cluster_{c}'] = [max_class_percentage, dominant_class, gini, min_IOU, max_IOU, mean_IOU, IQR_IOU]

        if v == True:
            print(f'Gini impurity of cluster {c}: {gini:.2f}')
            print(f'Dominant class in the cluster {c}: {dominant_class}')
            print(f'Percentage of points in the dominant class {dominant_class}: {max_class_percentage:.2f}%')
            print(f'ModIOU < 0.5: {len(cluster_labels[cluster_labels==0])}, ModIOU > 0.5: {len(cluster_labels[cluster_labels==1])}')
            print(f'Samples: {total_points}/{total}, Recall over total{c}: {(total_points/total) * 100:.2f}%')
            print(f'IQR: {IQR_IOU}, Mean IOU: {mean_IOU}\n')
        
        if show_img is not None:
            images_cluster_c = show_img[cluster_pred == c]
            correct = images_cluster_c[cluster_labels == 1]
            incorrect = images_cluster_c[cluster_labels == 0]

            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(correct[0].squeeze())
            ax[0].set_title('Label')
            ax[1].imshow(correct[1].squeeze())
            ax[2].imshow(correct[2].squeeze())
            fig.suptitle(f'Label: 1.0, Cluster: {c}')
            plt.show()

            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(incorrect[0].squeeze())
            ax[1].imshow(incorrect[1].squeeze())
            ax[2].imshow(incorrect[2].squeeze())
            fig.suptitle(f'Label: 0.0, Cluster: {c}')
            plt.show()

    unique_preds = sorted(list(clusters))
    data_to_plot = []

    for value in unique_preds:
        corresponding_gt = [cluster_gt[i] for i, v in enumerate(cluster_pred) if v == value]
        data_to_plot.append(corresponding_gt)

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data_to_plot)

    for component in ['whiskers', 'caps', 'boxes','fliers', 'means']:
        plt.setp(bp[component], color='white')

    # Setting the x-axis labels
    ax.set_xticks([i + 1 for i, _ in enumerate(unique_preds)])
    ax.set_xticklabels([str(i) for i in unique_preds])
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Distribution of mod IOU values")
    ax.set_title("Distribution of mod IOU per cluster")
    plt.show()

    return stats

def compute_metrics_for_threshold(y_true, cluster_pred, threshold):
    y_binary = np.where(y_true >= threshold, 1, 0)
    return cluster_info(cluster_pred, y_binary)

def plot_cluster_vs_threshold(y_true, cluster_pred, num_th = 50):
    thresholds = np.linspace(0, 1, num_th)  
    num_clusters = len(set(cluster_pred))
    results = {f'cluster_{i}': {'max_class_percentages': [], 'dominant_classes': [], 'ginis': []} for i in range(num_clusters)}

    for threshold in thresholds:
        metrics = compute_metrics_for_threshold(y_true, cluster_pred, threshold)
        for cluster, values in metrics.items():
            results[cluster]['max_class_percentages'].append(values[0])
            results[cluster]['dominant_classes'].append(values[1])
            results[cluster]['ginis'].append(values[2])

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    for cluster in results.keys():
        ax[0].plot(thresholds, results[cluster]['max_class_percentages'], label=cluster)
        ax[1].plot(thresholds, results[cluster]['dominant_classes'], label=cluster)
        ax[2].plot(thresholds, results[cluster]['ginis'], label=cluster)

    ax[0].set_title('Max Class Percentage over Thresholds')
    ax[0].set_xlabel('Threshold')
    ax[0].set_ylabel('Max Class Percentage')
    ax[0].legend()

    ax[1].set_title('Dominant Class over Thresholds')
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('Dominant Class')
    ax[1].legend()

    ax[2].set_title('Gini Impurity over Thresholds')
    ax[2].set_xlabel('Threshold')
    ax[2].set_ylabel('Gini Impurity')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

def ss_cluster(data, ss_dict, image_ids, true_lbl, pred_lbl, num_images, name):
     # num images is the number of images per cluster 
    dir_path = f"/app/ai_experiments/Lockout/tmp"

    c_dict = {}
    for ss_name, list_img in ss_dict.items():
        c_dict[ss_name] = {'true_lbl': [], 'pred_c':[]}
        print(ss_name)
        fig, axs = plt.subplots(2, 3)
        for i, v in enumerate(list_img):
            idx = image_ids.index(v)
            lbl = true_lbl[idx]
            c = pred_lbl[idx]
            c_dict[ss_name]['true_lbl'].append(lbl)
            c_dict[ss_name]['pred_c'].append(c)

            img = data[idx][0].squeeze()
            axs[i//3, i%3].imshow(img)
            axs[i//3, i%3].set_title(f'Label: {lbl}, Pred: {c}', fontsize=8)
        plt.suptitle(ss_name)
        plt.show()
    print(c_dict)
    cluster_set = set(pred_lbl)
    for c in cluster_set:
        num_rows = np.ceil(num_images / 4).astype(int)
        fig, axs = plt.subplots(num_rows, 4)
        for i in range(num_images):
            cluster_data = data[(pred_lbl==c)]
            img = cluster_data[i][0].squeeze()
            if num_rows > 1:
                axs[i//4, i%4].imshow(img)
                axs[i//4, i%4].set_title(f'Label: {true_lbl[(pred_lbl==c)][i]}')
            else:
                axs[i].imshow(img)
                axs[i].set_title(f'Label: {true_lbl[(pred_lbl==c)][i]}', fontsize=8)

        img_path = f"{dir_path}/{name}_cluster_fig_{c}.png"

    plt.savefig(img_path)  
    plt.show()
    plt.close(fig)

    return c_dict

labeled_data_test = {
    'Horizontal Lines': ['64b013017fac0c705623f96a','64a835d2b99ecd3728a81ab0', '64a05adf43739a1f4ce7e65b', '64bd7699cbfc280a313fd13e', '64b7db365753baef810b2d94', '64b996bf57d3da7d810790c3'], 
    'Vertical Lines': ['64a628fa3161e6999fdd2753','649f5a8a01c0b43dc51312ee', '64ac53670b8c1167a536d790', '64a8aa864135b8d856d9c656', '649d05752ec66a693e924dc6', '64b6f63bff9522e99cb16491'], 
    'No Lines': ['64998020336e43729366eea8','64afeab146240c84130aed1f','64bed6c2cdba41a5523f9f6d','64ac6399232723cdae094c54','64998020336e43729366eea8', '64b74ccd8dbfa4bb65cbf62f'],
    'Oblique Lines': ['64c3eb8124c8a4284fd8ac17','64b09a10b857825fa025c1bb','64b09a24a96d9bb8d7cb72bb', '64af2abbd71bd52b6701c482', '64ac4bee6737447a2f444810', '64962bf19c6e97516d81a9ac'],
    'Medical device': ['649f7be1d67692b3ce95dcb4','64a6ff996389ae3770acf2f0','64c28c9e0a60f0c3f8935db2', '64b56706ab2708155c088c45', '64b2cc89619eb13adea8028e', '64b61a2c789423cd3ec7238b'],
    'Zoomed out': ['64974b3c26fcf283200d63e4','64a7140eb3f025e0cbc9ca6d', '64af5fc532a528bae7c96b33','64a788379d12a25f04966f72', '649e4bcdf504c99d2971347d', '64a11b6e453793e8701463b5']
}